#!/usr/bin/env python3
"""
修复了图像处理错误的YOLO检测文件管理系统主机端
修复了栈溢出和标签页切换问题
"""

import sys
import os
import json
import base64
import cv2
import numpy as np
import threading
import time
import shutil
import traceback
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import socket
import atexit
import gc

# PyQt5 导入
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QTextEdit, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, \
    QFileDialog, QMessageBox, QProgressBar, QSplitter, QHeaderView, QCheckBox, QTreeWidget, QTreeWidgetItem, QMenu, \
    QAction
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QUrl
from PyQt5.QtGui import QFont, QColor, QPalette, QBrush, QIcon, QPixmap, QImage


# ====================== 数据模型类 ======================
class DetectionData:
    """检测数据类"""

    def __init__(self, data_dict: Dict[str, Any]):
        self.id = data_dict.get('frame_id', 0)
        self.timestamp = data_dict.get('timestamp', '')
        self.client_ip = data_dict.get('client_ip', '')
        self.source_type = data_dict.get('source_type', '')
        self.detections = data_dict.get('detections', [])
        self.frame_info = data_dict.get('frame_info', {})
        self.stats = data_dict.get('stats', {})
        self.image_base64 = data_dict.get('image_base64', None)
        self.raw_data = data_dict
        self.decoded_image = None  # 解码后的图像
        self.annotated_image = None  # 标注后的图像

    def get_detection_count(self) -> int:
        return len(self.detections)

    def get_image_size(self) -> str:
        if self.frame_info:
            return f"{self.frame_info.get('width', 0)}x{self.frame_info.get('height', 0)}"
        return "N/A"

    def get_class_distribution(self) -> Dict[str, int]:
        distribution = defaultdict(int)
        for det in self.detections:
            class_name = det.get('class', 'unknown')
            distribution[class_name] += 1
        return dict(distribution)

    def has_image(self) -> bool:
        return self.image_base64 is not None

    def decode_image(self) -> Optional[np.ndarray]:
        """解码base64图像"""
        if self.decoded_image is not None:
            return self.decoded_image

        if not self.image_base64:
            return None

        try:
            # 解码图像
            img_data = base64.b64decode(self.image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            self.decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 强制清理内存
            del img_data, nparr
            gc.collect()

        except Exception as e:
            print(f"解码图像失败: {e}")
            self.decoded_image = None

        return self.decoded_image

    def create_annotated_image(self) -> Optional[np.ndarray]:
        """创建带标注的图像"""
        if self.annotated_image is not None:
            return self.annotated_image

        # 解码图像
        image = self.decode_image()
        if image is None:
            return None

        try:
            # 创建图像的深拷贝以避免修改原始数据
            self.annotated_image = image.copy()

            # 绘制检测框和标签
            for det in self.detections:
                bbox = det.get('bbox', {})
                if 'xmin' in bbox and 'ymin' in bbox and 'xmax' in bbox and 'ymax' in bbox:
                    # 生成颜色基于类别
                    class_name = det.get('class', 'unknown')
                    color = self._get_color(class_name)

                    # 确保颜色值是整数
                    color = tuple(map(int, color))

                    # 绘制边界框
                    cv2.rectangle(self.annotated_image,
                                  (bbox['xmin'], bbox['ymin']),
                                  (bbox['xmax'], bbox['ymax']),
                                  color, 2)

                    # 准备标签文本
                    label = f"{class_name}: {det.get('confidence', 0):.2f}"

                    # 计算文本大小
                    font_scale = 0.5
                    thickness = 1
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    # 绘制标签背景
                    label_bg_top_left = (bbox['xmin'], max(0, bbox['ymin'] - label_height - 10))
                    label_bg_bottom_right = (bbox['xmin'] + label_width, bbox['ymin'])

                    # 绘制背景矩形
                    cv2.rectangle(self.annotated_image,
                                  label_bg_top_left,
                                  label_bg_bottom_right,
                                  color, cv2.FILLED)

                    # 绘制标签文本
                    text_y = max(bbox['ymin'] - 5, label_height + 5)
                    cv2.putText(self.annotated_image, label, (bbox['xmin'], text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            return self.annotated_image
        except Exception as e:
            print(f"创建标注图像失败: {e}")
            # 如果标注失败，返回原始图像
            return image
        finally:
            # 清理临时变量
            gc.collect()

    def _get_color(self, class_name: str) -> tuple:
        """根据类别名称获取颜色"""
        # 使用hash生成确定性颜色
        hash_value = hash(class_name) & 0xFFFFFF  # 限制在24位

        # 使用hash值生成RGB颜色
        r = (hash_value & 0xFF0000) >> 16
        g = (hash_value & 0x00FF00) >> 8
        b = hash_value & 0x0000FF

        # 确保颜色不太暗
        r = max(r, 100)
        g = max(g, 100)
        b = max(b, 100)

        return (b, g, r)  # OpenCV使用BGR格式

    def save_image(self, save_dir: str, prefix: str = "") -> Optional[str]:
        """保存图像到文件"""
        if not save_dir:
            return None

        try:
            # 确保保存目录存在
            os.makedirs(save_dir, exist_ok=True)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{prefix}detection_{timestamp}.jpg" if prefix else f"detection_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)

            # 尝试保存标注图像
            annotated_image = self.create_annotated_image()
            if annotated_image is not None:
                cv2.imwrite(filepath, annotated_image)
                return filepath

            # 如果标注图像不存在，尝试保存原始图像
            original_image = self.decode_image()
            if original_image is not None:
                cv2.imwrite(filepath, original_image)
                return filepath

            return None
        except Exception as e:
            print(f"保存图像失败: {e}")
            return None
        finally:
            gc.collect()


class FileManager:
    """文件管理器"""

    def __init__(self, base_dir: str = ""):
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images") if base_dir else ""
        self.detections_dir = os.path.join(base_dir, "detections") if base_dir else ""
        self.stats_dir = os.path.join(base_dir, "stats") if base_dir else ""

        # 创建必要的目录
        if base_dir:
            self._create_dirs()

    def _create_dirs(self):
        """创建必要的目录"""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.detections_dir, exist_ok=True)
            os.makedirs(self.stats_dir, exist_ok=True)
        except Exception as e:
            print(f"创建目录失败: {e}")

    def set_base_dir(self, base_dir: str):
        """设置基础目录"""
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.detections_dir = os.path.join(base_dir, "detections")
        self.stats_dir = os.path.join(base_dir, "stats")
        self._create_dirs()

    def save_detection(self, detection: DetectionData) -> Dict[str, Any]:
        """保存检测数据"""
        try:
            result = {"status": "success", "saved_files": []}

            # 保存图像
            if detection.has_image():
                image_path = detection.save_image(self.images_dir)
                if image_path:
                    result["saved_files"].append(image_path)

            # 保存检测数据
            detection_path = os.path.join(self.detections_dir,
                                          f"detection_{detection.timestamp.replace(':', '-')}_{detection.id}.json")
            with open(detection_path, 'w', encoding='utf-8') as f:
                json.dump(detection.raw_data, f, indent=2, ensure_ascii=False)
            result["saved_files"].append(detection_path)

            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_file_list(self, file_type: str = "all") -> List[Dict[str, Any]]:
        """获取文件列表"""
        if not self.base_dir or not os.path.exists(self.base_dir):
            return []

        try:
            files = []

            # 确定搜索路径
            search_paths = []
            if file_type in ["all", "images"]:
                search_paths.append(self.images_dir)
            if file_type in ["all", "detections"]:
                search_paths.append(self.detections_dir)
            if file_type in ["all", "stats"]:
                search_paths.append(self.stats_dir)

            for search_path in search_paths:
                if not os.path.exists(search_path):
                    continue

                for root, dirs, filenames in os.walk(search_path):
                    for filename in filenames:
                        filepath = os.path.join(root, filename)
                        try:
                            stat = os.stat(filepath)
                            file_type = "image" if filename.lower().endswith(
                                ('.jpg', '.jpeg', '.png', '.bmp')) else "detection" if filename.lower().endswith(
                                '.json') else "stats"

                            files.append({
                                "name": filename,
                                "path": filepath,
                                "type": file_type,
                                "size": stat.st_size,
                                "created": datetime.fromtimestamp(stat.st_ctime),
                                "modified": datetime.fromtimestamp(stat.st_mtime)
                            })
                        except:
                            continue

            # 按修改时间排序（最新的在前面）
            files.sort(key=lambda x: x["modified"], reverse=True)
            return files
        except Exception as e:
            print(f"获取文件列表失败: {e}")
            return []

    def get_directory_size(self) -> Dict[str, Any]:
        """获取目录大小"""
        if not self.base_dir or not os.path.exists(self.base_dir):
            return {"total_size": 0, "total_size_mb": 0}

        total_size = 0
        file_count = 0

        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                try:
                    filepath = os.path.join(root, file)
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except:
                    continue

        return {
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count
        }

    def delete_files(self, filepaths: List[str]) -> Dict[str, Any]:
        """删除文件"""
        result = {
            "success": 0,
            "failed": 0,
            "failed_files": []
        }

        for filepath in filepaths:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    result["success"] += 1
                else:
                    result["failed"] += 1
                    result["failed_files"].append(f"文件不存在: {filepath}")
            except Exception as e:
                result["failed"] += 1
                result["failed_files"].append(f"{filepath}: {str(e)}")

        return result

    def cleanup_old_files(self, days: int = 30) -> Dict[str, Any]:
        """清理旧文件"""
        cutoff_time = datetime.now() - timedelta(days=days)

        result = {
            "deleted": 0,
            "errors": []
        }

        try:
            for root, dirs, files in os.walk(self.base_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if mtime < cutoff_time:
                            os.remove(filepath)
                            result["deleted"] += 1
                    except Exception as e:
                        result["errors"].append(f"删除 {filepath} 失败: {str(e)}")
        except Exception as e:
            result["errors"].append(f"遍历目录失败: {str(e)}")

        return result


# ====================== 图像显示工作线程 ======================
class ImageDisplayWorker(QThread):
    """图像显示工作线程 - 修复内存泄漏版本"""

    image_ready = pyqtSignal(QImage, object)
    image_failed = pyqtSignal(str)

    def __init__(self, detection_data):
        super().__init__()
        self.detection_data = detection_data
        self._should_stop = False
        # 设置更小的栈大小防止溢出
        self.setStackSize(256 * 1024)

    def stop(self):
        """停止线程"""
        self._should_stop = True
        if self.isRunning():
            self.quit()
            self.wait(100)

    def run(self):
        """运行线程"""
        try:
            # 检查是否需要停止
            if self._should_stop:
                return

            # 解码图像并添加标注
            annotated_image = self.detection_data.create_annotated_image()
            if annotated_image is not None and not self._should_stop:
                # 转换BGR到RGB
                rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                # 创建QImage
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w

                # 确保图像数据是连续的
                if not rgb_image.flags['C_CONTIGUOUS']:
                    rgb_image = np.ascontiguousarray(rgb_image)

                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # 创建深拷贝并发射信号
                q_image_copy = q_image.copy()

                # 清理内存
                del rgb_image
                gc.collect()

                # 发射信号
                self.image_ready.emit(q_image_copy, self.detection_data)

        except Exception as e:
            print(f"图像处理错误: {e}")
            self.image_failed.emit(str(e))
        finally:
            # 清理内存
            gc.collect()


# ====================== 服务器线程类 ======================
class ServerThread(QThread):
    """服务器线程"""

    # 定义信号
    log_signal = pyqtSignal(str, str)
    status_signal = pyqtSignal(dict)
    detection_signal = pyqtSignal(dict)
    detection_image_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    server_stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.host = "0.0.0.0"
        self.port = 5000
        self.save_dir = ""
        self.running = False
        self.server = None
        self.file_manager = None
        self._should_stop = False

        # 数据存储
        self.detections = deque(maxlen=1000)
        self.statistics = {
            'total_frames': 0,
            'total_detections': 0,
            'connected_clients': set(),
            'start_time': time.time()
        }

    def configure(self, host: str, port: int, save_dir: str):
        """配置服务器"""
        self.host = host
        self.port = port
        self.save_dir = save_dir
        self.file_manager = FileManager(save_dir)

    def run(self):
        """运行服务器线程"""
        try:
            # 创建Flask应用
            from flask import Flask, request, jsonify
            from flask_cors import CORS
            import warnings
            warnings.filterwarnings("ignore", message="This is a development server")

            # 创建Flask应用
            app = Flask(__name__)
            CORS(app)

            # 更新状态
            self.running = True
            self._should_stop = False
            self.log_signal.emit(f"服务器启动在 {self.host}:{self.port}", "info")
            self.status_signal.emit({"status": "starting", "host": self.host, "port": self.port})

            # 定义路由
            @app.route('/api/upload', methods=['POST'])
            def upload_result():
                try:
                    if self._should_stop:
                        return jsonify({"status": "error", "message": "服务器正在关闭"}), 503

                    data = request.json
                    client_ip = request.remote_addr

                    # 验证数据
                    if not data or 'detections' not in data:
                        return jsonify({"status": "error", "message": "无效的数据格式"}), 400

                    # 创建检测数据对象
                    data['client_ip'] = client_ip
                    detection = DetectionData(data)

                    # 更新统计
                    self.statistics['total_frames'] += 1
                    self.statistics['total_detections'] += detection.get_detection_count()
                    self.statistics['connected_clients'].add(client_ip)

                    # 保存数据
                    save_result = None
                    if self.file_manager:
                        save_result = self.file_manager.save_detection(detection)

                    # 添加到内存存储
                    self.detections.append(detection)

                    # 发送检测数据信号到GUI
                    self.detection_signal.emit({
                        'id': detection.id,
                        'timestamp': detection.timestamp,
                        'client_ip': detection.client_ip,
                        'detection_count': detection.get_detection_count(),
                        'image_size': detection.get_image_size(),
                        'save_result': save_result
                    })

                    # 发送检测图像信号到GUI
                    self.detection_image_signal.emit(detection)

                    # 更新状态
                    self.status_signal.emit({
                        'status': 'running',
                        'total_frames': self.statistics['total_frames'],
                        'total_detections': self.statistics['total_detections'],
                        'connected_clients': list(self.statistics['connected_clients']),
                        'uptime': time.time() - self.statistics['start_time']
                    })

                    return jsonify({
                        "status": "success",
                        "message": f"接收成功，检测到 {detection.get_detection_count()} 个对象",
                        "received_time": datetime.now().isoformat()
                    })

                except Exception as e:
                    self.log_signal.emit(f"处理上传错误: {str(e)}", "error")
                    return jsonify({"status": "error", "message": str(e)}), 500

            @app.route('/api/health', methods=['GET'])
            def health_check():
                return jsonify({
                    "status": "running",
                    "timestamp": datetime.now().isoformat(),
                    "server": f"{self.host}:{self.port}",
                    "statistics": {
                        "total_frames": self.statistics['total_frames'],
                        "total_detections": self.statistics['total_detections'],
                        "connected_clients": list(self.statistics['connected_clients'])
                    }
                })

            @app.route('/api/stats', methods=['GET'])
            def get_stats():
                return jsonify({
                    "status": "success",
                    "statistics": {
                        "total_frames": self.statistics['total_frames'],
                        "total_detections": self.statistics['total_detections'],
                        "connected_clients": list(self.statistics['connected_clients']),
                        "uptime_seconds": time.time() - self.statistics['start_time']
                    }
                })

            @app.route('/api/shutdown', methods=['POST'])
            def shutdown_server():
                """用于安全关闭服务器的路由"""
                self._should_stop = True
                return jsonify({"status": "success", "message": "正在关闭服务器"})

            # 使用Werkzeug开发服务器
            from werkzeug.serving import make_server

            # 创建服务器
            self.server = make_server(self.host, self.port, app, threaded=True)

            self.log_signal.emit("服务器已启动，等待连接...", "info")
            self.status_signal.emit({"status": "running"})

            # 设置超时，使服务器可以检查停止标志
            self.server.timeout = 1

            # 运行服务器
            while not self._should_stop:
                self.server.handle_request()

            self.log_signal.emit("服务器正在关闭...", "info")

        except Exception as e:
            self.log_signal.emit(f"服务器启动失败: {str(e)}", "error")
            self.error_signal.emit(str(e))
            traceback.print_exc()
        finally:
            self.running = False
            self._should_stop = True
            self.server_stopped.emit()
            self.log_signal.emit("服务器已停止", "info")

    def stop(self):
        """停止服务器"""
        if not self.running:
            return

        self.log_signal.emit("正在停止服务器...", "info")
        self._should_stop = True

        # 发送关闭请求
        try:
            import requests
            requests.post(f"http://{self.host}:{self.port}/api/shutdown", timeout=1)
        except:
            pass

        # 等待线程结束
        self.wait(2000)


# ====================== 工作线程类 ======================
class WorkerSignals(QObject):
    """工作线程的信号"""
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    file_table_ready = pyqtSignal(list)
    test_result_ready = pyqtSignal(str, bool)
    disk_usage_ready = pyqtSignal(str)
    sys_info_ready = pyqtSignal(str)
    delete_result_ready = pyqtSignal(dict)
    cleanup_result_ready = pyqtSignal(dict)


class Worker(QRunnable):
    """通用工作线程"""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit((type(e), e, e.__traceback__))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


# ====================== 主窗口类 ======================
class MainWindow(QMainWindow):
    """主窗口"""

    # 定义信号
    file_table_updated = pyqtSignal(list)
    test_result_received = pyqtSignal(str, bool)
    disk_usage_updated = pyqtSignal(str)
    sys_info_updated = pyqtSignal(str)
    delete_result_received = pyqtSignal(dict)
    cleanup_result_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.server_thread = None
        self.file_manager = FileManager()
        self.current_files = []
        self.is_stopping_server = False
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)  # 限制最大线程数

        self.image_workers = []  # 图像工作线程列表
        self.display_fps = 10  # 显示帧率
        self.last_frame_time = 0  # 上一帧时间
        self.frame_count = 0  # 接收到的帧数
        self.show_image = True  # 是否显示图像
        self.current_detection_data = None  # 当前检测数据
        self.current_qimage = None  # 当前QImage

        self.image_update_timer = QTimer()  # 图像更新定时器
        self.image_update_timer.timeout.connect(self.process_pending_images)
        self.pending_images = deque(maxlen=5)  # 待处理的图像队列，限制大小

        # 用于跟踪标签页状态
        self.is_display_tab_active = False
        self.last_display_update = 0

        # 初始化UI
        self.init_ui()

        # 设置窗口属性
        self.setWindowTitle("YOLO检测文件管理系统 - 主机端")
        self.setGeometry(100, 100, 1400, 900)

        # 应用样式
        self.apply_stylesheet()

        # 连接信号
        self.connect_signals()

        # 设置图像更新定时器
        self.image_update_timer.start(50)  # 每50ms检查一次

    def connect_signals(self):
        """连接信号"""
        self.file_table_updated.connect(self.update_file_table)
        self.test_result_received.connect(self.show_test_result)
        self.disk_usage_updated.connect(self.set_disk_usage_label)
        self.sys_info_updated.connect(self.set_sys_info_text)
        self.delete_result_received.connect(self.on_files_deleted)
        self.cleanup_result_received.connect(self.on_cleanup_completed)

    def init_ui(self):
        """初始化用户界面"""
        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)  # 连接标签页切换信号
        main_layout.addWidget(self.tab_widget)

        # 创建各个标签页
        self.create_connection_tab()
        self.create_file_management_tab()
        self.create_monitoring_tab()
        self.create_detection_display_tab()
        self.create_settings_tab()

        # 创建状态栏
        self.create_status_bar()

    def on_tab_changed(self, index):
        """标签页切换时调用"""
        tab_text = self.tab_widget.tabText(index)
        self.is_display_tab_active = (tab_text == "检测画面")

        # 如果切换到检测画面标签页，更新显示
        if self.is_display_tab_active and self.current_detection_data:
            self.update_current_image_display()

    def create_connection_tab(self):
        """创建连接配置标签页"""
        connection_tab = QWidget()
        layout = QVBoxLayout(connection_tab)

        # 服务器配置组
        server_group = QGroupBox("服务器配置")
        server_layout = QVBoxLayout()

        # 主机IP
        host_layout = QHBoxLayout()
        host_layout.addWidget(QLabel("主机IP:"))
        self.host_input = QLineEdit("192.168.0.112")
        host_layout.addWidget(self.host_input)
        server_layout.addLayout(host_layout)

        # 端口
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("端口:"))
        self.port_input = QLineEdit("5000")
        port_layout.addWidget(self.port_input)
        server_layout.addLayout(port_layout)

        server_group.setLayout(server_layout)
        layout.addWidget(server_group)

        # 保存路径组
        save_group = QGroupBox("保存路径配置")
        save_layout = QVBoxLayout()

        # 路径选择
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit(r"F:\pi-server\pidata")
        path_layout.addWidget(self.path_input)

        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.browse_save_path)
        path_layout.addWidget(self.browse_button)

        save_layout.addLayout(path_layout)

        # 路径信息
        self.path_info_label = QLabel("未选择保存路径")
        self.path_info_label.setWordWrap(True)
        save_layout.addWidget(self.path_info_label)

        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        # 服务器控制按钮
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("启动服务器")
        self.start_button.clicked.connect(self.start_server)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("停止服务器")
        self.stop_button.clicked.connect(self.stop_server)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        button_layout.addWidget(self.stop_button)

        self.test_button = QPushButton("测试连接")
        self.test_button.clicked.connect(self.test_connection)
        button_layout.addWidget(self.test_button)

        layout.addLayout(button_layout)

        # 服务器状态
        status_group = QGroupBox("服务器状态")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("状态: 未启动")
        status_layout.addWidget(self.status_label)

        self.connection_label = QLabel("连接数: 0")
        status_layout.addWidget(self.connection_label)

        self.frame_label = QLabel("接收帧数: 0")
        status_layout.addWidget(self.frame_label)

        self.detection_label = QLabel("检测总数: 0")
        status_layout.addWidget(self.detection_label)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 添加弹性空间
        layout.addStretch()

        self.tab_widget.addTab(connection_tab, "连接配置")

    def create_file_management_tab(self):
        """创建文件管理标签页"""
        file_tab = QWidget()
        layout = QVBoxLayout(file_tab)

        # 文件操作工具栏
        toolbar_layout = QHBoxLayout()

        self.refresh_button = QPushButton("刷新列表")
        self.refresh_button.clicked.connect(self.refresh_file_list)
        toolbar_layout.addWidget(self.refresh_button)

        self.open_folder_button = QPushButton("打开文件夹")
        self.open_folder_button.clicked.connect(self.open_save_folder)
        toolbar_layout.addWidget(self.open_folder_button)

        self.delete_button = QPushButton("删除选中")
        self.delete_button.clicked.connect(self.delete_selected_files)
        self.delete_button.setStyleSheet("background-color: #ff9800; color: white;")
        toolbar_layout.addWidget(self.delete_button)

        self.cleanup_button = QPushButton("清理旧文件")
        self.cleanup_button.clicked.connect(self.cleanup_old_files)
        toolbar_layout.addWidget(self.cleanup_button)

        toolbar_layout.addStretch()

        # 文件类型过滤
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("文件类型:"))

        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["全部文件", "图像文件", "检测文件", "统计文件"])
        self.file_type_combo.currentTextChanged.connect(self.refresh_file_list)
        filter_layout.addWidget(self.file_type_combo)

        # 保留天数设置
        filter_layout.addWidget(QLabel("保留天数:"))
        self.retention_spin = QSpinBox()
        self.retention_spin.setRange(1, 365)
        self.retention_spin.setValue(30)
        filter_layout.addWidget(self.retention_spin)

        filter_layout.addStretch()

        layout.addLayout(toolbar_layout)
        layout.addLayout(filter_layout)

        # 文件表格
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(6)
        self.file_table.setHorizontalHeaderLabels(["选择", "文件名", "类型", "大小", "修改时间", "路径"])

        # 设置表格属性
        self.file_table.horizontalHeader().setStretchLastSection(True)
        self.file_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.file_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # 添加复选框列
        self.file_table.setColumnWidth(0, 50)
        self.file_table.setColumnWidth(1, 200)
        self.file_table.setColumnWidth(2, 80)
        self.file_table.setColumnWidth(3, 100)
        self.file_table.setColumnWidth(4, 150)

        layout.addWidget(self.file_table)

        # 选中文件信息
        info_group = QGroupBox("文件信息")
        info_layout = QVBoxLayout()

        self.selected_file_info = QTextEdit()
        self.selected_file_info.setReadOnly(True)
        self.selected_file_info.setMaximumHeight(100)
        info_layout.addWidget(self.selected_file_info)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # 连接表格选择变化信号
        self.file_table.itemSelectionChanged.connect(self.update_file_info)

        self.tab_widget.addTab(file_tab, "文件管理")

    def create_monitoring_tab(self):
        """创建监控标签页"""
        monitor_tab = QWidget()
        layout = QVBoxLayout(monitor_tab)

        # 实时数据监控
        realtime_group = QGroupBox("实时数据监控")
        realtime_layout = QVBoxLayout()

        # 数据表格
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(["ID", "时间", "客户端IP", "检测数", "图像尺寸", "状态"])

        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)

        realtime_layout.addWidget(self.data_table)

        realtime_group.setLayout(realtime_layout)
        layout.addWidget(realtime_group)

        # 统计信息
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        self.tab_widget.addTab(monitor_tab, "实时监控")

    def create_detection_display_tab(self):
        """创建检测显示标签页"""
        display_tab = QWidget()
        layout = QVBoxLayout(display_tab)

        # 控制面板
        control_group = QGroupBox("显示控制")
        control_layout = QVBoxLayout()

        # 显示模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("显示模式:"))

        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["原始图像", "标注图像"])
        self.display_mode_combo.setCurrentIndex(1)  # 默认显示标注图像
        self.display_mode_combo.currentIndexChanged.connect(self.on_display_mode_changed)
        mode_layout.addWidget(self.display_mode_combo)

        # 显示控制
        display_control_layout = QHBoxLayout()

        self.show_image_check = QCheckBox("实时显示图像")
        self.show_image_check.setChecked(True)
        self.show_image_check.stateChanged.connect(self.on_show_image_changed)
        display_control_layout.addWidget(self.show_image_check)

        display_control_layout.addStretch()

        # FPS控制
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("显示帧率:"))

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self.on_fps_changed)
        fps_layout.addWidget(self.fps_spin)

        fps_layout.addStretch()

        control_layout.addLayout(mode_layout)
        control_layout.addLayout(display_control_layout)
        control_layout.addLayout(fps_layout)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 图像显示区域
        image_group = QGroupBox("实时检测画面")
        image_layout = QVBoxLayout()

        # 创建滚动区域用于显示图像
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        # 创建图像显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setText("等待接收图像...")
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")

        self.scroll_area.setWidget(self.image_label)
        image_layout.addWidget(self.scroll_area)

        # 图像信息
        info_layout = QHBoxLayout()

        self.image_info_label = QLabel("图像信息: 未接收")
        info_layout.addWidget(self.image_info_label)

        info_layout.addStretch()

        # 保存按钮
        self.save_image_button = QPushButton("保存当前图像")
        self.save_image_button.clicked.connect(self.save_current_image)
        self.save_image_button.setEnabled(False)
        info_layout.addWidget(self.save_image_button)

        image_layout.addLayout(info_layout)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        self.tab_widget.addTab(display_tab, "检测画面")

    def create_settings_tab(self):
        """创建设置标签页"""
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)

        # 保存设置
        save_group = QGroupBox("保存设置")
        save_layout = QVBoxLayout()

        # 自动保存
        auto_save_layout = QHBoxLayout()
        self.auto_save_check = QCheckBox("自动保存接收的数据")
        self.auto_save_check.setChecked(True)
        auto_save_layout.addWidget(self.auto_save_check)
        save_layout.addLayout(auto_save_layout)

        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        # 显示设置
        display_group = QGroupBox("显示设置")
        display_layout = QVBoxLayout()

        # 刷新频率
        refresh_layout = QHBoxLayout()
        refresh_layout.addWidget(QLabel("刷新频率 (Hz):"))
        self.refresh_rate_spin = QSpinBox()
        self.refresh_rate_spin.setRange(1, 60)
        self.refresh_rate_spin.setValue(10)
        refresh_layout.addWidget(self.refresh_rate_spin)
        display_layout.addLayout(refresh_layout)

        # 最大显示行数
        max_rows_layout = QHBoxLayout()
        max_rows_layout.addWidget(QLabel("最大显示行数:"))
        self.max_rows_spin = QSpinBox()
        self.max_rows_spin.setRange(10, 1000)
        self.max_rows_spin.setValue(100)
        max_rows_layout.addWidget(self.max_rows_spin)
        display_layout.addLayout(max_rows_layout)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # 系统信息
        sys_group = QGroupBox("系统信息")
        sys_layout = QVBoxLayout()

        self.sys_info_text = QTextEdit()
        self.sys_info_text.setReadOnly(True)
        self.sys_info_text.setMaximumHeight(100)
        sys_layout.addWidget(self.sys_info_text)

        # 更新系统信息按钮
        update_info_btn = QPushButton("更新系统信息")
        update_info_btn.clicked.connect(self.update_system_info)
        sys_layout.addWidget(update_info_btn)

        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)

        # 添加弹性空间
        layout.addStretch()

        self.tab_widget.addTab(settings_tab, "系统设置")

    def create_status_bar(self):
        """创建状态栏"""
        self.statusBar().showMessage("就绪")

        # 添加状态标签
        self.server_status_label = QLabel("服务器: 停止")
        self.statusBar().addPermanentWidget(self.server_status_label)

        self.file_count_label = QLabel("文件数: 0")
        self.statusBar().addPermanentWidget(self.file_count_label)

        self.disk_usage_label = QLabel("磁盘使用: 0 MB")
        self.statusBar().addPermanentWidget(self.disk_usage_label)

        # 添加图像状态标签
        self.image_status_label = QLabel("图像: 未接收")
        self.statusBar().addPermanentWidget(self.image_status_label)

    def apply_stylesheet(self):
        """应用样式表"""
        style = """
        QMainWindow {
            background-color: #f5f5f5;
        }

        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }

        QTableWidget {
            background-color: white;
            alternate-background-color: #f8f8f8;
            selection-background-color: #4CAF50;
            selection-color: white;
            gridline-color: #ddd;
        }

        QTableWidget::item {
            padding: 5px;
        }

        QHeaderView::section {
            background-color: #4CAF50;
            color: white;
            padding: 5px;
            border: 1px solid #388E3C;
            font-weight: bold;
        }

        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #1976D2;
        }

        QPushButton:pressed {
            background-color: #0D47A1;
        }

        QPushButton:disabled {
            background-color: #BDBDBD;
            color: #757575;
        }

        QLineEdit, QComboBox, QSpinBox {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 3px;
            background-color: white;
        }

        QTextEdit {
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 3px;
            font-family: Consolas, monospace;
        }

        QLabel {
            color: #333;
        }

        QScrollArea {
            border: 1px solid #ccc;
            background-color: white;
        }
        """
        self.setStyleSheet(style)

    # ====================== 槽函数 ======================

    def browse_save_path(self):
        """浏览保存路径"""
        path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.path_input.text())
        if path:
            self.path_input.setText(path)
            self.update_path_info(path)

    def update_path_info(self, path):
        """更新路径信息"""
        if os.path.exists(path):
            # 计算目录大小
            def calculate_size():
                total_size = 0
                file_count = 0
                for root, dirs, files in os.walk(path):
                    for file in files:
                        try:
                            total_size += os.path.getsize(os.path.join(root, file))
                            file_count += 1
                        except:
                            pass
                return path, file_count, total_size

            # 在工作线程中计算
            worker = Worker(calculate_size)
            worker.signals.result.connect(lambda result: self.update_path_info_callback(*result))
            self.threadpool.start(worker)

    def update_path_info_callback(self, path, file_count, total_size):
        """更新路径信息回调"""
        self.path_info_label.setText(
            f"路径: {path}\n"
            f"文件数: {file_count}\n"
            f"总大小: {round(total_size / (1024 * 1024), 2)} MB"
        )

        # 更新文件管理器
        self.file_manager.set_base_dir(path)
        self.refresh_file_list()

    def start_server(self):
        """启动服务器"""
        if self.is_stopping_server:
            QMessageBox.warning(self, "警告", "服务器正在停止中，请稍后再试")
            return

        # 获取配置
        host = self.host_input.text().strip()
        port = self.port_input.text().strip()
        save_dir = self.path_input.text().strip()

        # 验证输入
        if not host:
            QMessageBox.warning(self, "警告", "请输入主机IP地址")
            return

        if not port.isdigit():
            QMessageBox.warning(self, "警告", "端口号必须是数字")
            return

        if not save_dir:
            QMessageBox.warning(self, "警告", "请选择保存路径")
            return

        # 如果服务器已经在运行，先停止
        if self.server_thread and self.server_thread.isRunning():
            QMessageBox.warning(self, "警告", "服务器已经在运行中")
            return

        # 创建并启动服务器线程
        self.server_thread = ServerThread()
        self.server_thread.configure(host, int(port), save_dir)

        # 连接信号
        self.server_thread.log_signal.connect(self.handle_log)
        self.server_thread.status_signal.connect(self.handle_status)
        self.server_thread.detection_signal.connect(self.handle_detection)
        self.server_thread.detection_image_signal.connect(self.handle_detection_image)
        self.server_thread.error_signal.connect(self.handle_error)
        self.server_thread.server_stopped.connect(self.on_server_stopped)
        self.server_thread.finished.connect(self.on_server_finished)

        # 启动线程
        self.server_thread.start()

        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"状态: 启动中 ({host}:{port})")

    def stop_server(self):
        """停止服务器"""
        if not self.server_thread or not self.server_thread.isRunning():
            return

        if self.is_stopping_server:
            return

        self.is_stopping_server = True
        self.stop_button.setEnabled(False)
        self.status_label.setText("状态: 正在停止...")

        # 停止所有图像处理线程
        self.stop_all_image_workers()

        # 在工作线程中执行停止操作
        worker = Worker(self.perform_server_stop)
        self.threadpool.start(worker)

    def stop_all_image_workers(self):
        """停止所有图像处理线程"""
        for worker in self.image_workers[:]:  # 使用切片创建副本
            try:
                if worker.isRunning():
                    worker.stop()
            except:
                pass
        self.image_workers.clear()
        self.pending_images.clear()

    def perform_server_stop(self):
        """执行服务器停止操作"""
        try:
            self.server_thread.stop()
        except Exception as e:
            print(f"停止服务器时出错: {str(e)}")

    def on_server_stopped(self):
        """服务器已停止信号处理"""
        pass

    def on_server_finished(self):
        """服务器线程完成信号处理"""
        self.is_stopping_server = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("状态: 已停止")
        self.server_status_label.setText("服务器: 停止")
        self.image_status_label.setText("图像: 服务器已停止")

        # 清理内存
        gc.collect()

    def test_connection(self):
        """测试连接"""
        host = self.host_input.text().strip()
        port = self.port_input.text().strip()

        if not host or not port:
            QMessageBox.warning(self, "警告", "请输入主机IP和端口")
            return

        # 在工作线程中测试连接
        def test_connection_task(host):
            import subprocess
            try:
                # Windows ping命令
                param = '-n' if os.name == 'nt' else '-c'
                result = subprocess.run(['ping', param, '1', host],
                                        capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    return f"成功连接到 {host}", True
                else:
                    return f"无法连接到 {host}", False
            except subprocess.TimeoutExpired:
                return "连接测试超时", False
            except Exception as e:
                return f"连接测试失败: {str(e)}", False

        worker = Worker(test_connection_task, host)
        worker.signals.result.connect(lambda result: self.test_result_received.emit(*result))
        self.threadpool.start(worker)

    @pyqtSlot(str, bool)
    def show_test_result(self, message: str, success: bool):
        """显示测试结果"""
        if success:
            QMessageBox.information(self, "连接测试", message)
        else:
            QMessageBox.warning(self, "连接测试", message)

    def handle_log(self, message: str, level: str = "info"):
        """处理日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"

        if level == "error":
            self.statusBar().showMessage(f"错误: {message}", 5000)
        elif level == "warning":
            self.statusBar().showMessage(f"警告: {message}", 3000)
        else:
            self.statusBar().showMessage(message, 2000)

        # 更新日志显示（如果有的话）
        print(f"{level.upper()}: {formatted_msg}")

    def handle_status(self, status_data: dict):
        """处理状态更新"""
        status = status_data.get('status', 'unknown')

        if status == 'starting':
            self.server_status_label.setText("服务器: 启动中")
        elif status == 'running':
            self.server_status_label.setText("服务器: 运行中")

            # 更新连接信息
            total_frames = status_data.get('total_frames', 0)
            total_detections = status_data.get('total_detections', 0)
            connected_clients = status_data.get('connected_clients', [])
            uptime = status_data.get('uptime', 0)

            self.status_label.setText(f"状态: 运行中")
            self.connection_label.setText(f"连接数: {len(connected_clients)}")
            self.frame_label.setText(f"接收帧数: {total_frames}")
            self.detection_label.setText(f"检测总数: {total_detections}")

            # 更新状态栏
            self.statusBar().showMessage(f"服务器运行中 | 连接: {len(connected_clients)} | 帧: {total_frames}", 1000)

    def handle_detection(self, detection_data: dict):
        """处理检测数据"""
        # 添加到数据表格
        self.data_table.insertRow(0)  # 插入到顶部

        # 设置数据
        self.data_table.setItem(0, 0, QTableWidgetItem(str(detection_data.get('id', ''))))
        self.data_table.setItem(0, 1, QTableWidgetItem(detection_data.get('timestamp', '')))
        self.data_table.setItem(0, 2, QTableWidgetItem(detection_data.get('client_ip', '')))
        self.data_table.setItem(0, 3, QTableWidgetItem(str(detection_data.get('detection_count', 0))))
        self.data_table.setItem(0, 4, QTableWidgetItem(detection_data.get('image_size', 'N/A')))

        save_result = detection_data.get('save_result', {})
        if save_result.get('status') == 'success':
            self.data_table.setItem(0, 5, QTableWidgetItem("已保存"))
        else:
            self.data_table.setItem(0, 5, QTableWidgetItem("未保存"))

        # 限制行数
        max_rows = self.max_rows_spin.value()
        while self.data_table.rowCount() > max_rows:
            self.data_table.removeRow(self.data_table.rowCount() - 1)

        # 更新统计信息
        self.update_stats_display()

    def handle_detection_image(self, detection_data):
        """处理检测图像数据"""
        self.frame_count += 1

        # 更新图像状态
        self.image_status_label.setText(f"图像: 接收 {self.frame_count} 帧")

        # 存储当前检测数据
        self.current_detection_data = detection_data

        # 检查是否应该显示图像
        if not self.show_image or not self.is_display_tab_active:
            return

        # 检查帧率限制
        current_time = time.time()
        if current_time - self.last_frame_time < 1.0 / self.display_fps:
            return  # 跳过此帧以维持目标帧率

        self.last_frame_time = current_time

        # 将图像添加到待处理队列
        self.pending_images.append(detection_data)

    def process_pending_images(self):
        """处理待显示的图像"""
        if not self.pending_images or not self.is_display_tab_active:
            return

        # 获取最新的图像
        try:
            detection_data = self.pending_images.pop()
        except IndexError:
            return

        # 在工作线程中处理图像
        worker = ImageDisplayWorker(detection_data)
        worker.image_ready.connect(self.update_detection_image)
        worker.image_failed.connect(lambda e: print(f"图像处理失败: {e}"))
        worker.finished.connect(lambda: self.cleanup_image_worker(worker))
        worker.start()

        # 将工作线程添加到列表以便管理
        self.image_workers.append(worker)

    def cleanup_image_worker(self, worker):
        """清理图像工作线程"""
        try:
            if worker in self.image_workers:
                self.image_workers.remove(worker)
            worker.deleteLater()
        except:
            pass
        finally:
            gc.collect()

    def update_current_image_display(self):
        """更新当前图像显示（用于标签页切换时）"""
        if not self.is_display_tab_active or not self.current_detection_data:
            return

        # 在工作线程中处理图像
        worker = ImageDisplayWorker(self.current_detection_data)
        worker.image_ready.connect(self.update_detection_image)
        worker.image_failed.connect(lambda e: print(f"图像处理失败: {e}"))
        worker.finished.connect(lambda: self.cleanup_image_worker(worker))
        worker.start()

        # 将工作线程添加到列表以便管理
        self.image_workers.append(worker)

    def update_detection_image(self, qimage, detection_data):
        """更新检测图像显示"""
        if not self.show_image or not self.is_display_tab_active:
            return

        try:
            self.current_qimage = qimage

            # 根据显示模式显示图像
            display_mode = self.display_mode_combo.currentText()

            if display_mode == "原始图像":
                # 显示原始图像
                self.show_original_image(detection_data)
            else:  # 标注图像
                # 显示图像
                pixmap = QPixmap.fromImage(qimage)
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)

            # 更新图像信息
            self.update_image_info(detection_data)

            # 启用保存按钮
            self.save_image_button.setEnabled(True)

        except Exception as e:
            print(f"更新图像显示失败: {e}")

    def show_original_image(self, detection_data):
        """显示原始图像"""
        # 获取原始图像
        original_img = detection_data.decode_image()
        if original_img is not None:
            # 转换BGR到RGB
            rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w

            # 确保图像数据是连续的
            if not rgb_img.flags['C_CONTIGUOUS']:
                rgb_img = np.ascontiguousarray(rgb_img)

            original_qimage = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 显示图像
            pixmap = QPixmap.fromImage(original_qimage)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

            # 清理内存
            del rgb_img, original_img
            gc.collect()
        else:
            # 如果没有原始图像，显示占位符
            self.image_label.setText("无原始图像数据")

    def update_image_info(self, detection_data):
        """更新图像信息"""
        info = f"帧ID: {detection_data.id} | "
        info += f"时间: {detection_data.timestamp} | "
        info += f"客户端: {detection_data.client_ip} | "
        info += f"检测数: {detection_data.get_detection_count()} | "
        info += f"尺寸: {detection_data.get_image_size()}"

        self.image_info_label.setText(f"图像信息: {info}")

    def on_display_mode_changed(self, index):
        """显示模式改变"""
        # 更新当前显示的图像
        if self.current_detection_data and self.is_display_tab_active:
            # 重新处理图像
            worker = ImageDisplayWorker(self.current_detection_data)
            worker.image_ready.connect(self.update_detection_image)
            worker.image_failed.connect(lambda e: print(f"图像处理失败: {e}"))
            worker.finished.connect(lambda: self.cleanup_image_worker(worker))
            worker.start()

    def on_show_image_changed(self, state):
        """显示图像复选框状态改变"""
        self.show_image = (state == Qt.Checked)

        if not self.show_image:
            self.image_label.clear()
            self.image_label.setText("图像显示已关闭")
            self.image_info_label.setText("图像信息: 显示已关闭")
            self.save_image_button.setEnabled(False)
        elif self.current_detection_data and self.is_display_tab_active:
            # 重新显示图像
            worker = ImageDisplayWorker(self.current_detection_data)
            worker.image_ready.connect(self.update_detection_image)
            worker.image_failed.connect(lambda e: print(f"图像处理失败: {e}"))
            worker.finished.connect(lambda: self.cleanup_image_worker(worker))
            worker.start()

    def on_fps_changed(self, fps):
        """FPS设置改变"""
        self.display_fps = fps

    def save_current_image(self):
        """保存当前图像"""
        if not self.current_detection_data or not self.file_manager.base_dir:
            QMessageBox.warning(self, "警告", "没有可保存的图像或未设置保存路径")
            return

        try:
            # 保存图像
            save_result = self.current_detection_data.save_image(self.file_manager.images_dir, prefix="manual_")

            if save_result:
                QMessageBox.information(self, "保存成功", f"图像已保存到:\n{save_result}")
            else:
                QMessageBox.warning(self, "保存失败", "无法保存图像")

        except Exception as e:
            QMessageBox.critical(self, "保存错误", f"保存图像时出错: {str(e)}")

    def handle_error(self, error_message: str):
        """处理错误"""
        self.handle_log(f"服务器错误: {error_message}", "error")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("状态: 错误")
        self.is_stopping_server = False

    def refresh_file_list(self):
        """刷新文件列表"""
        if not self.file_manager.base_dir:
            return

        # 在工作线程中执行文件列表刷新
        def get_file_list():
            # 获取文件类型过滤
            file_type_text = self.file_type_combo.currentText()
            file_type_map = {
                "全部文件": "all",
                "图像文件": "images",
                "检测文件": "detections",
                "统计文件": "stats"
            }
            file_type = file_type_map.get(file_type_text, "all")

            # 获取文件列表
            return self.file_manager.get_file_list(file_type)

        worker = Worker(get_file_list)
        worker.signals.result.connect(lambda result: self.file_table_updated.emit(result))
        self.threadpool.start(worker)

    @pyqtSlot(list)
    def update_file_table(self, files):
        """更新文件表格"""
        self.current_files = files

        # 清空表格
        self.file_table.setRowCount(0)

        # 填充表格
        for i, file_info in enumerate(self.current_files):
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)

            # 复选框
            checkbox = QTableWidgetItem()
            checkbox.setCheckState(Qt.Unchecked)
            self.file_table.setItem(row, 0, checkbox)

            # 文件名
            self.file_table.setItem(row, 1, QTableWidgetItem(file_info['name']))

            # 文件类型
            type_icon = "📷" if file_info['type'] == 'image' else "📄" if file_info['type'] == 'detection' else "📊"
            self.file_table.setItem(row, 2, QTableWidgetItem(f"{type_icon} {file_info['type']}"))

            # 文件大小
            size_mb = file_info['size'] / (1024 * 1024)
            size_text = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_info['size'] / 1024:.2f} KB"
            self.file_table.setItem(row, 3, QTableWidgetItem(size_text))

            # 修改时间
            self.file_table.setItem(row, 4, QTableWidgetItem(file_info['modified'].strftime("%Y-%m-%d %H:%M:%S")))

            # 路径
            self.file_table.setItem(row, 5, QTableWidgetItem(file_info['path']))

        # 更新状态栏
        self.file_count_label.setText(f"文件数: {len(self.current_files)}")

        # 更新磁盘使用
        self.update_disk_usage()

    def update_file_info(self):
        """更新选中文件信息"""
        selected_items = self.file_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        if row < len(self.current_files):
            file_info = self.current_files[row]

            info_text = f"""
文件名: {file_info['name']}
路径: {file_info['path']}
类型: {file_info['type']}
大小: {file_info['size']} 字节 ({file_info['size'] / (1024 * 1024):.2f} MB)
创建时间: {file_info['created'].strftime("%Y-%m-%d %H:%M:%S")}
修改时间: {file_info['modified'].strftime("%Y-%m-%d %H:%M:%S")}
            """
            self.selected_file_info.setText(info_text.strip())

    def open_save_folder(self):
        """打开保存文件夹"""
        if self.file_manager.base_dir and os.path.exists(self.file_manager.base_dir):
            # 在工作线程中执行
            def open_folder():
                import subprocess
                import platform

                try:
                    if platform.system() == "Windows":
                        os.startfile(self.file_manager.base_dir)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.Popen(["open", self.file_manager.base_dir])
                    else:  # Linux
                        subprocess.Popen(["xdg-open", self.file_manager.base_dir])
                    return True, None
                except Exception as e:
                    return False, str(e)

            worker = Worker(open_folder)
            worker.signals.result.connect(lambda result: self.on_open_folder_result(*result))
            self.threadpool.start(worker)
        else:
            QMessageBox.warning(self, "警告", "保存路径不存在")

    def on_open_folder_result(self, success, error_message):
        """打开文件夹结果处理"""
        if not success:
            QMessageBox.warning(self, "警告", f"无法打开文件夹: {error_message}")

    def delete_selected_files(self):
        """删除选中的文件"""
        # 获取选中的文件
        selected_files = []
        for row in range(self.file_table.rowCount()):
            checkbox = self.file_table.item(row, 0)
            if checkbox and checkbox.checkState() == Qt.Checked:
                filepath = self.file_table.item(row, 5).text()
                selected_files.append(filepath)

        if not selected_files:
            QMessageBox.warning(self, "警告", "请先选择要删除的文件")
            return

        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除选中的 {len(selected_files)} 个文件吗？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 在工作线程中执行删除操作
            def delete_files():
                return self.file_manager.delete_files(selected_files)

            worker = Worker(delete_files)
            worker.signals.result.connect(lambda result: self.delete_result_received.emit(result))
            self.threadpool.start(worker)

    @pyqtSlot(dict)
    def on_files_deleted(self, result):
        """文件删除完成回调"""
        if result['success'] > 0:
            QMessageBox.information(self, "删除完成",
                                    f"成功删除 {result['success']} 个文件\n"
                                    f"失败 {result['failed']} 个")

            # 刷新文件列表
            self.refresh_file_list()

        if result['failed'] > 0:
            QMessageBox.warning(self, "删除失败",
                                f"以下文件删除失败:\n" + "\n".join(result['failed_files'][:10]))

    def cleanup_old_files(self):
        """清理旧文件"""
        days = self.retention_spin.value()

        reply = QMessageBox.question(
            self, "确认清理",
            f"确定要清理 {days} 天前的旧文件吗？\n此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 在工作线程中执行清理操作
            def cleanup_files():
                return self.file_manager.cleanup_old_files(days)

            worker = Worker(cleanup_files)
            worker.signals.result.connect(lambda result: self.cleanup_result_received.emit(result))
            self.threadpool.start(worker)

    @pyqtSlot(dict)
    def on_cleanup_completed(self, result):
        """清理完成回调"""
        if result['deleted'] > 0:
            QMessageBox.information(self, "清理完成",
                                    f"成功清理 {result['deleted']} 个旧文件")

            # 刷新文件列表
            self.refresh_file_list()

        if result['errors']:
            QMessageBox.warning(self, "清理错误",
                                f"清理过程中发生错误:\n" + "\n".join(result['errors'][:5]))

    def update_stats_display(self):
        """更新统计信息显示"""
        if self.server_thread:
            stats_text = f"""
服务器统计:
==============
总接收帧数: {self.server_thread.statistics['total_frames']}
总检测数: {self.server_thread.statistics['total_detections']}
连接客户端数: {len(self.server_thread.statistics['connected_clients'])}
运行时间: {time.time() - self.server_thread.statistics['start_time']:.1f} 秒
            """
            self.stats_text.setText(stats_text.strip())

    def update_disk_usage(self):
        """更新磁盘使用信息"""
        if self.file_manager.base_dir:
            # 在工作线程中执行
            def get_disk_usage():
                size_info = self.file_manager.get_directory_size()
                return f"磁盘使用: {size_info['total_size_mb']} MB"

            worker = Worker(get_disk_usage)
            worker.signals.result.connect(lambda result: self.disk_usage_updated.emit(result))
            self.threadpool.start(worker)

    @pyqtSlot(str)
    def set_disk_usage_label(self, text: str):
        """设置磁盘使用标签"""
        self.disk_usage_label.setText(text)

    def update_system_info(self):
        """更新系统信息"""

        # 在工作线程中执行
        def get_system_info():
            try:
                import platform
                import psutil

                # 获取系统信息
                sys_info = f"""
系统信息:
==============
操作系统: {platform.system()} {platform.release()}
Python版本: {platform.python_version()}
处理器: {platform.processor()}

内存使用:
总内存: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB
已使用: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB
使用率: {psutil.virtual_memory().percent}%

磁盘信息:
总空间: {psutil.disk_usage('/').total / (1024 ** 3):.2f} GB
可用空间: {psutil.disk_usage('/').free / (1024 ** 3):.2f} GB
使用率: {psutil.disk_usage('/').percent}%
                """
                return sys_info.strip()
            except Exception as e:
                return f"获取系统信息失败: {str(e)}"

        worker = Worker(get_system_info)
        worker.signals.result.connect(lambda result: self.sys_info_updated.emit(result))
        self.threadpool.start(worker)

    @pyqtSlot(str)
    def set_sys_info_text(self, text: str):
        """设置系统信息文本"""
        self.sys_info_text.setText(text)

    def closeEvent(self, event):
        """关闭事件"""
        try:
            # 停止所有图像处理线程
            self.stop_all_image_workers()

            # 停止图像更新定时器
            self.image_update_timer.stop()

            # 停止服务器线程
            if self.server_thread and self.server_thread.isRunning():
                reply = QMessageBox.question(
                    self, "确认退出",
                    "服务器仍在运行，确定要退出吗？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self.stop_server()
                    # 等待服务器停止
                    QTimer.singleShot(1000, event.accept)
                    event.ignore()
                    return
                else:
                    event.ignore()
                    return

            # 清理线程池
            self.threadpool.clear()
            self.threadpool.waitForDone(3000)

            # 强制垃圾回收
            gc.collect()

            event.accept()

        except Exception as e:
            print(f"关闭时发生错误: {e}")
            event.accept()


# ====================== 应用程序入口 ======================
def main():
    """主函数"""
    # 设置递归限制防止栈溢出
    sys.setrecursionlimit(10000)

    # 创建应用程序
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO-PCB缺陷检测文件管理系统 - 主机端")

    # 设置异常处理
    def exception_hook(exctype, value, traceback_obj):
        """处理未捕获的异常"""
        print(f"未捕获的异常: {exctype.__name__}: {value}")
        if traceback_obj:
            traceback.print_tb(traceback_obj)

        # 显示错误消息
        QMessageBox.critical(None, "错误",
                             f"程序发生未捕获的异常:\n\n{exctype.__name__}: {value}\n\n程序将退出。")

        # 退出程序
        sys.exit(1)

    sys.excepthook = exception_hook

    # 创建并显示主窗口
    try:
        window = MainWindow()
        window.show()

        # 运行应用程序
        result = app.exec_()

        # 退出前清理
        gc.collect()

        sys.exit(result)

    except Exception as e:
        print(f"启动应用程序时发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()