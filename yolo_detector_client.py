#!/usr/bin/env python3
"""
YOLO Detection Client - Raspberry Pi Side
Supports picamera, usb camera, image files and other input sources
"""

import os
import sys
import argparse
import glob
import time
import json
import base64
import requests
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from typing import List, Dict, Any, Tuple
import threading
from queue import Queue, Empty
import socket


# ====================== Configuration Class ======================
class ClientConfig:
    """Client Configuration"""

    def __init__(self):
        self.server_url = "http://192.168.0.112:5000"  # Default server address
        self.send_interval = 0  # Send interval (seconds), 0 means send every frame
        self.send_with_image = True  # Whether to send images
        self.send_threshold = 0  # Detection threshold, only send when exceed this number
        self.retry_count = 3  # Retry count
        self.timeout = 5  # Timeout (seconds)
        self.batch_size = 1  # Batch send size
        self.enable_local_display = True  # Whether to display locally
        self.save_locally = False  # Whether to save locally
        self.compress_quality = 80  # Image compression quality (1-100)
        self.max_queue_size = 100  # Maximum queue size
        self.local_ip = "192.168.0.123"  # Raspberry Pi IP

    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        if args.server:
            self.server_url = args.server
        if args.no_image:
            self.send_with_image = False
        if args.interval:
            self.send_interval = args.interval
        if args.threshold:
            self.send_threshold = args.threshold
        if args.batch_size:
            self.batch_size = args.batch_size
        if args.compress:
            self.compress_quality = args.compress
        if args.no_display:
            self.enable_local_display = False


# ====================== Communication Class ======================
class DetectionSender:
    """Detection Result Sender"""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.queue = Queue(maxsize=config.max_queue_size)
        self.sending = False
        self.stats = {
            "total_sent": 0,
            "total_failed": 0,
            "queue_size": 0,
            "last_success_time": None,
            "last_error": None,
            "connection_status": "disconnected"
        }

    def start_sender_thread(self):
        """Start sender thread"""
        self.sending = True
        self.sender_thread = threading.Thread(target=self._send_worker, daemon=True)
        self.sender_thread.start()
        print(f"Sender thread started, server: {self.config.server_url}")

    def stop_sender(self):
        """Stop sender"""
        self.sending = False
        if hasattr(self, 'sender_thread'):
            self.sender_thread.join(timeout=2)

    def add_to_queue(self, data: Dict[str, Any]):
        """Add data to send queue"""
        try:
            self.queue.put_nowait(data)
            self.stats["queue_size"] = self.queue.qsize()
        except:
            # Queue full, discard oldest data
            try:
                self.queue.get_nowait()  # Remove oldest data
                self.queue.put_nowait(data)  # Add new data
                self.stats["queue_size"] = self.queue.qsize()
                print("Queue full, discarding oldest data")
            except:
                pass

    def _send_worker(self):
        """Send worker thread"""
        batch = []
        last_send_time = 0

        while self.sending:
            try:
                current_time = time.time()

                # Get data from queue
                try:
                    data = self.queue.get(timeout=0.1)
                    batch.append(data)

                    # Check if batch size reached or need immediate send
                    force_send = (len(batch) >= self.config.batch_size)

                    # Check send interval
                    time_to_send = (self.config.send_interval > 0 and
                                    current_time - last_send_time >= self.config.send_interval)

                    if force_send or time_to_send or self.config.send_interval == 0:
                        if batch:
                            self._send_batch(batch)
                            batch = []
                            last_send_time = current_time

                except Empty:
                    # Queue empty, check if there's pending batch
                    if batch and self.config.send_interval > 0:
                        if current_time - last_send_time >= self.config.send_interval:
                            self._send_batch(batch)
                            batch = []
                            last_send_time = current_time

            except Exception as e:
                self.stats["last_error"] = str(e)
                time.sleep(0.5)

        # Send remaining data before exiting
        if batch:
            self._send_batch(batch)

    def _send_batch(self, batch: List[Dict[str, Any]]):
        """Send batch data"""
        for data in batch:
            success = self._send_single(data)
            if success:
                self.stats["total_sent"] += 1
                self.stats["last_success_time"] = datetime.now().isoformat()
                self.stats["connection_status"] = "connected"
            else:
                self.stats["total_failed"] += 1
                self.stats["connection_status"] = "disconnected"

    def _send_single(self, data: Dict[str, Any]) -> bool:
        """Send single detection result"""
        for attempt in range(self.config.retry_count):
            try:
                response = requests.post(
                    f"{self.config.server_url}/api/upload",
                    json=data,
                    headers={'Content-Type': 'application/json'},
                    timeout=self.config.timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        return True
                    else:
                        print(f"Server returned error: {result.get('message')}")
                else:
                    print(f"HTTP error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                if attempt == self.config.retry_count - 1:
                    print(f"Cannot connect to server: {self.config.server_url}")
            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1}/{self.config.retry_count})")
            except Exception as e:
                print(f"Send error: {str(e)} (attempt {attempt + 1}/{self.config.retry_count})")

            time.sleep(1)  # Wait before retry

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get sender statistics"""
        self.stats["queue_size"] = self.queue.qsize()
        return self.stats.copy()


# ====================== YOLO Detector Class ======================
class YOLODetector:
    """YOLO Detector"""

    def __init__(self, model_path: str, confidence_thresh: float = 0.5):
        self.model_path = model_path
        self.confidence_thresh = confidence_thresh

        # Load model
        print(f"Loading YOLO model: {model_path}")
        try:
            self.model = YOLO(model_path, task='detect')
            self.labels = self.model.names
            print(f"Model loaded, total {len(self.labels)} classes")
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)

    def detect_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray, float]:
        """Detect single frame"""
        start_time = time.time()

        # Run inference
        results = self.model(frame, verbose=False)

        # Extract detection results
        detections = []
        annotated_frame = frame.copy()

        if len(results) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Get bounding box coordinates
                xyxy_tensor = boxes[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)

                # Get class and confidence
                class_id = int(boxes[i].cls.item())
                class_name = self.labels[class_id]
                confidence = boxes[i].conf.item()

                if confidence >= self.confidence_thresh:
                    # Add to detection results
                    detection = {
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": float(confidence),
                        "bbox": {
                            "xmin": int(xmin),
                            "ymin": int(ymin),
                            "xmax": int(xmax),
                            "ymax": int(ymax),
                            "width": int(xmax - xmin),
                            "height": int(ymax - ymin),
                            "center_x": int((xmin + xmax) / 2),
                            "center_y": int((ymin + ymax) / 2)
                        }
                    }
                    detections.append(detection)

                    # Draw bounding box on image
                    color = self._get_color(class_id)
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), color, 2)

                    # Add label
                    label = f'{class_name}: {confidence:.2f}'
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, label_size[1] + 10)

                    # Label background
                    cv2.rectangle(annotated_frame,
                                  (xmin, label_ymin - label_size[1] - 10),
                                  (xmin + label_size[0], label_ymin + base_line - 10),
                                  color, cv2.FILLED)

                    # Label text
                    cv2.putText(annotated_frame, label, (xmin, label_ymin - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        inference_time = time.time() - start_time
        return detections, annotated_frame, inference_time

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color based on class ID"""
        colors = [
            (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
            (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184),
            (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)
        ]
        return colors[class_id % len(colors)]


# ====================== Main Application Class ======================
class YOLOClientApp:
    """YOLO Client Application"""

    def __init__(self, args):
        self.args = args
        self.config = ClientConfig()
        self.config.update_from_args(args)

        # Initialize components
        self.detector = YOLODetector(args.model, float(args.thresh))
        self.sender = DetectionSender(self.config)

        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.total_inference_time = 0
        self.start_time = time.time()

        # Determine source type
        self.source_type = self._determine_source_type(args.source)
        print(f"Input source type: {self.source_type}")

        # Get local IP
        self._get_local_ip()

    def _get_local_ip(self):
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.local_ip = s.getsockname()[0]
            s.close()
            print(f"Local IP address: {self.local_ip}")
        except:
            self.local_ip = self.config.local_ip
            print(f"Using configured IP: {self.local_ip}")

    def _determine_source_type(self, source: str) -> str:
        """Determine input source type"""
        img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv', '.flv', '.AVI', '.MOV', '.MP4', '.MKV', '.WMV', '.FLV']

        if os.path.isdir(source):
            return 'folder'
        elif os.path.isfile(source):
            _, ext = os.path.splitext(source)
            if ext.lower() in img_ext_list:
                return 'image'
            elif ext.lower() in vid_ext_list:
                return 'video'
            else:
                print(f'Unsupported extension: {ext}')
                sys.exit(1)
        elif 'usb' in source.lower():
            return 'usb'
        elif source.lower() == 'libcamera':
            return 'libcamera'
        # ===== Recognize picamera source =====
        elif source.lower().startswith('picamera'):
            return 'picamera'
        # ===== End recognition =====
        else:
            print(f'Invalid input source: {source}')
            sys.exit(1)

    def _connect_to_server(self) -> bool:
        """Connect to server"""
        print(f"Trying to connect to server: {self.config.server_url}")
        try:
            response = requests.get(f"{self.config.server_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Successfully connected to server")
                print(f"  Server status: {data.get('status', 'unknown')}")
                print(f"  Server: {data.get('server', 'unknown')}")

                # Get statistics if available
                stats = data.get('statistics', {})
                print(f"  Total frames: {stats.get('total_frames', 0)}")
                print(f"  Total detections: {stats.get('total_detections', 0)}")
                print(f"  Connected clients: {len(stats.get('connected_clients', []))}")

                # Get uptime from server if available
                uptime_seconds = stats.get('uptime_seconds', 0)
                if uptime_seconds > 0:
                    uptime_hours = uptime_seconds / 3600
                    print(f"  Server uptime: {uptime_hours:.1f} hours")
                return True
        except Exception as e:
            print(f"✗ Failed to connect to server: {str(e)}")
            print("Please check:")
            print(f"  1. Is server address correct: {self.config.server_url}")
            print(f"  2. Is server running")
            print(f"  3. Is network connection normal")
        return False

    def _prepare_data_packet(self, frame: np.ndarray,
                             detections: List[Dict[str, Any]],
                             source_type: str,
                             inference_time: float) -> Dict[str, Any]:
        """Prepare data packet for sending"""
        data = {
            "frame_id": self.frame_count,
            "timestamp": datetime.now().isoformat(),
            "source_type": source_type,
            "detections": detections,
            "client_ip": self.local_ip,
            "frame_info": {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1
            },
            "stats": {
                "object_count": len(detections),
                "inference_time": inference_time,
                "total_frames": self.frame_count,
                "total_detections": self.total_detections,
                "avg_inference_time": self.total_inference_time / self.frame_count if self.frame_count > 0 else 0
            }
        }

        # If image needs to be sent
        if self.config.send_with_image:
            # Compress image to reduce transmission size
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.compress_quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            data["image_base64"] = img_base64

        return data

    def _should_send(self, detections: List[Dict[str, Any]]) -> bool:
        """Determine whether to send"""
        # Based on detection count
        if self.config.send_threshold > 0 and len(detections) < self.config.send_threshold:
            return False

        # Based on send interval
        if self.config.send_interval > 0:
            if hasattr(self, 'last_send_time'):
                if time.time() - self.last_send_time < self.config.send_interval:
                    return False
            self.last_send_time = time.time()

        return True

    def _display_stats(self, frame: np.ndarray, fps: float):
        """Display statistics"""
        height, width = frame.shape[:2]

        # Top status bar background
        cv2.rectangle(frame, (0, 0), (width, 110), (0, 0, 0), -1)

        # Server status
        sender_stats = self.sender.get_stats()
        status_color = (0, 255, 0) if sender_stats["connection_status"] == "connected" else (0, 0, 255)
        status_text = f"Status: {sender_stats['connection_status']}"
        cv2.putText(frame, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # FPS and detection info
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f'Frames: {self.frame_count}', (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f'Detections: {self.total_detections}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Right side statistics
        avg_inference = self.total_inference_time / self.frame_count if self.frame_count > 0 else 0
        cv2.putText(frame, f'Inference: {avg_inference * 1000:.1f}ms', (width - 200, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        queue_size = sender_stats["queue_size"]
        cv2.putText(frame, f'Queue: {queue_size}', (width - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        sent_count = sender_stats["total_sent"]
        cv2.putText(frame, f'Sent: {sent_count}', (width - 200, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Bottom info bar
        cv2.rectangle(frame, (0, height - 30), (width, height), (0, 0, 0), -1)
        server_addr = self.config.server_url.split("//")[
            -1] if "//" in self.config.server_url else self.config.server_url
        cv2.putText(frame, f'Raspberry Pi: {self.local_ip} | Server: {server_addr}',
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _print_status(self):
        """Print status to console"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_inference = self.total_inference_time / self.frame_count if self.frame_count > 0 else 0

        sender_stats = self.sender.get_stats()

        print(f"\rFrames: {self.frame_count:5d} | "
              f"FPS: {fps:5.1f} | "
              f"Detections: {self.total_detections:5d} | "
              f"Inference: {avg_inference * 1000:5.1f}ms | "
              f"Sent: {sender_stats['total_sent']:4d}({sender_stats['total_failed']:2d}) | "
              f"Queue: {sender_stats['queue_size']:3d} | "
              f"Status: {sender_stats['connection_status']:12s}", end="")

    def run(self):
        """Run main application"""
        # Connect to server
        if not self._connect_to_server():
            choice = input("Continue in offline mode? (y/n): ")
            if choice.lower() != 'y':
                return

        # Start sender thread
        self.sender.start_sender_thread()

        # Initialize based on source type
        if self.source_type in ['video', 'usb', 'libcamera', 'picamera']:
            self._run_realtime()
        else:
            self._run_batch()

        # Cleanup
        self._cleanup()

    def _run_realtime(self):
        """Run real-time detection"""
        # Initialize video capture
        cap = None
        picam2 = None

        if self.source_type == 'video':
            cap = cv2.VideoCapture(self.args.source)
        elif self.source_type == 'usb':
            camera_index = 0
            if self.args.source.lower().startswith('usb'):
                try:
                    camera_index = int(self.args.source[3:])
                except:
                    pass
            cap = cv2.VideoCapture(camera_index)
        elif self.source_type == 'libcamera':
            width, height = 640, 480
            if self.args.resolution:
                resW, resH = int(self.args.resolution.split('x')[0]), int(self.args.resolution.split('x')[1])
                width, height = resW, resH

            gst_pipeline = f'libcamerasrc ! video/x-raw, width={width}, height={height} ! videoconvert ! appsink'
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            if not cap.isOpened():
                print("Cannot open CSI camera, trying normal camera...")
                cap = cv2.VideoCapture(0)
        elif self.source_type == 'picamera':
            print(f"Initializing Picamera2...")
            try:
                from picamera2 import Picamera2

                # Parse resolution
                width, height = 640, 480
                if self.args.resolution:
                    resW, resH = int(self.args.resolution.split('x')[0]), int(self.args.resolution.split('x')[1])
                    width, height = resW, resH
                else:
                    print(f"No resolution specified, using default {width}x{height}")

                # Create and configure Picamera2
                picam2 = Picamera2()
                preview_config = picam2.create_preview_configuration(
                    main={"size": (width, height), "format": "RGB888"}
                )
                picam2.configure(preview_config)
                picam2.start()
                time.sleep(2)  # Wait for camera to stabilize
                print(f"✓ Picamera2 started successfully, resolution: {width}x{height}")

            except ImportError:
                print("Error: picamera2 library not found, please run: pip install picamera2")
                return
            except Exception as e:
                print(f"Failed to initialize Picamera2: {e}")
                return

        if cap is None and picam2 is None and self.source_type != 'picamera':
            print(f"Cannot open video source: {self.args.source}")
            return

        # FPS calculation
        fps_buffer = []
        fps_avg_len = 30
        last_status_time = time.time()

        print(f"\nStarting real-time detection...")
        print("Press 'q' to quit, 's' to pause, 'p' to save screenshot, 'i' to show detailed info")
        print("-" * 100)

        while True:
            t_start = time.time()

            # Read frame
            frame = None
            if self.source_type in ['video', 'usb', 'libcamera']:
                ret, frame = cap.read()
                if not ret:
                    print("Cannot read frame, exiting...")
                    break
            elif self.source_type == 'picamera' and picam2:
                try:
                    # Capture image array from Picamera2
                    frame_array = picam2.capture_array()

                    # Ensure image is 3-channel BGR format
                    if frame_array.shape[2] == 4:  # BGRA format
                        frame = cv2.cvtColor(frame_array, cv2.COLOR_BGRA2BGR)
                    elif frame_array.shape[2] == 3:  # RGB format
                        frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    else:
                        frame = frame_array

                except Exception as e:
                    print(f"Failed to capture frame from Picamera2: {e}")
                    break

            if frame is None:
                print("Frame is empty, exiting...")
                break

            # Adjust resolution
            if self.args.resolution and self.source_type != 'picamera':
                resW, resH = int(self.args.resolution.split('x')[0]), int(self.args.resolution.split('x')[1])
                frame = cv2.resize(frame, (resW, resH))

            # Detection
            detections, annotated_frame, inference_time = self.detector.detect_frame(frame)

            # Update statistics
            self.frame_count += 1
            self.total_detections += len(detections)
            self.total_inference_time += inference_time

            # Prepare and send data
            if self._should_send(detections):
                data = self._prepare_data_packet(frame, detections, self.source_type, inference_time)
                self.sender.add_to_queue(data)

            # Calculate FPS
            t_end = time.time()
            fps = 1.0 / (t_end - t_start)
            fps_buffer.append(fps)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_fps = np.mean(fps_buffer) if fps_buffer else 0

            # Display statistics
            if self.config.enable_local_display:
                self._display_stats(annotated_frame, avg_fps)
                cv2.imshow('YOLO Detection - Raspberry Pi', annotated_frame)

            # Periodically print status
            if time.time() - last_status_time > 1.0:
                self._print_status()
                last_status_time = time.time()

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("\nPaused, press any key to continue...")
                cv2.waitKey(0)
            elif key == ord('p'):
                filename = f'capture_{self.frame_count}.jpg'
                cv2.imwrite(filename, annotated_frame)
                print(f"\nScreenshot saved: {filename}")
            elif key == ord('i'):
                # Show detailed info
                print(f"\nDetailed info:")
                print(f"  Current frame detections: {len(detections)}")
                print(f"  Inference time: {inference_time * 1000:.1f}ms")
                print(f"  Current FPS: {fps:.1f}")
                print(f"  Average FPS: {avg_fps:.1f}")

        # Cleanup resources
        if self.source_type in ['video', 'usb', 'libcamera'] and cap is not None:
            cap.release()
        elif self.source_type == 'picamera' and picam2 is not None:
            picam2.stop()
            print("Picamera2 stopped")

        if self.config.enable_local_display:
            cv2.destroyAllWindows()

    def _run_batch(self):
        """Run batch processing"""
        # Get image list
        if self.source_type == 'image':
            image_files = [self.args.source]
        else:  # folder
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                pattern = os.path.join(self.args.source, '*' + ext)
                image_files.extend(glob.glob(pattern))

        print(f"Found {len(image_files)} images")

        for i, img_file in enumerate(image_files):
            print(f"\nProcessing image {i + 1}/{len(image_files)}: {os.path.basename(img_file)}")

            # Read image
            frame = cv2.imread(img_file)
            if frame is None:
                print(f"Cannot read image: {img_file}")
                continue

            # Adjust resolution
            if self.args.resolution:
                resW, resH = int(self.args.resolution.split('x')[0]), int(self.args.resolution.split('x')[1])
                frame = cv2.resize(frame, (resW, resH))

            # Detection
            detections, annotated_frame, inference_time = self.detector.detect_frame(frame)

            # Update statistics
            self.frame_count += 1
            self.total_detections += len(detections)
            self.total_inference_time += inference_time

            print(f"Detected {len(detections)} objects, inference time: {inference_time * 1000:.1f}ms")

            # Prepare and send data
            if self._should_send(detections):
                data = self._prepare_data_packet(frame, detections, self.source_type, inference_time)
                self.sender.add_to_queue(data)

            # Display results
            if self.config.enable_local_display:
                cv2.putText(annotated_frame, f'Detections: {len(detections)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f'File: {os.path.basename(img_file)}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow('YOLO Detection', annotated_frame)

                key = cv2.waitKey(0)
                if key == ord('q'):
                    break

        if self.config.enable_local_display:
            cv2.destroyAllWindows()

    def _cleanup(self):
        """Cleanup resources"""
        print("\n" + "=" * 80)
        print("Detection completed, statistics:")
        print("=" * 80)

        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_inference = self.total_inference_time / self.frame_count if self.frame_count > 0 else 0

        print(f"  Total frames: {self.frame_count}")
        print(f"  Total detections: {self.total_detections}")
        print(f"  Running time: {elapsed_time:.1f} seconds")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Average inference time: {avg_inference * 1000:.1f}ms")

        sender_stats = self.sender.get_stats()
        print(f"\nSend statistics:")
        print(f"  Successfully sent: {sender_stats['total_sent']}")
        print(f"  Failed to send: {sender_stats['total_failed']}")
        success_rate = (
                    sender_stats['total_sent'] / (sender_stats['total_sent'] + sender_stats['total_failed']) * 100) if (
                                                                                                                                   sender_stats[
                                                                                                                                       'total_sent'] +
                                                                                                                                   sender_stats[
                                                                                                                                       'total_failed']) > 0 else 0
        print(f"  Success rate: {success_rate:.1f}%")
        if sender_stats['last_success_time']:
            print(f"  Last success: {sender_stats['last_success_time']}")

        # Stop sender
        self.sender.stop_sender()
        print("=" * 80)


# ====================== Main Function ======================
def main():
    parser = argparse.ArgumentParser(
        description='YOLO Detection Client - Raspberry Pi Side (Supports Picamera2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --model yolov8n.pt --source picamera --server http://192.168.0.112:5000
  python %(prog)s --model yolov8n.pt --source picamera --resolution 640x480 --server http://192.168.0.112:5000
  python %(prog)s --model yolov8n.pt --source usb0 --server http://192.168.0.112:5000
  python %(prog)s --model yolov8n.pt --source test.jpg --server http://192.168.0.112:5000
        """
    )

    # Required arguments
    parser.add_argument('--model', required=True, help='YOLO model file path')
    parser.add_argument('--source', required=True,
                        help='Input source: image file, folder, video file, usb0, libcamera, picamera')
    parser.add_argument('--server', required=True, help='Server address, e.g.: http://192.168.0.112:5000')

    # Optional arguments
    parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--resolution', help='Display resolution WxH (e.g.: 640x480)')
    parser.add_argument('--interval', type=float, help='Send interval (seconds), 0 means send every frame')
    parser.add_argument('--threshold', type=int, help='Minimum detection threshold, only send when exceed')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch send size')
    parser.add_argument('--compress', type=int, default=80, choices=range(1, 101),
                        help='Image compression quality 1-100 (default: 80)')
    parser.add_argument('--no-image', action='store_true', help='Do not send image data, only detection results')
    parser.add_argument('--no-display', action='store_true', help='Do not display local window')
    parser.add_argument('--record', action='store_true', help='Record video')

    args = parser.parse_args()

    # Check model file
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        sys.exit(1)

    # Create and run application
    app = YOLOClientApp(args)

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        app._cleanup()
    except Exception as e:
        print(f"\nProgram error: {e}")
        import traceback
        traceback.print_exc()
        app._cleanup()


if __name__ == '__main__':
    main()