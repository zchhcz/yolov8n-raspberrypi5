<img width="1422" height="955" alt="image" src="https://github.com/user-attachments/assets/5e937bec-64b2-4c83-8ba8-24a2956c9042" /># yolov8n-raspberrypi5
基于yolov8n,树莓派5的PCB缺陷检测系统


这是基于yolov8n,树莓派5的PCB缺陷检测系统,从树莓派调用摄像头,经过树莓派推理后能将推理后的图片,数据等传输至局域网下的另一台主机,从而缓解工厂流水线高强度人力劳作等场景

这是本人上传至github的第一个项目,不喜勿喷,我有玉米症,叠个甲:

本科毕设级水准!
本科毕设级水准!
本科毕设级水准!

那就是水

各个方面都有很大的提升空间,参考了很多b站上的视频,部分命令或方式可能有雷同,若你有更好的想法或者改进欢迎联系我,邮箱:kexuehe1i@foxmail.com,如果对你有帮助的话请点一颗stars,非常感谢!

这是初始版本,下一个版本可能着重于提升模型精度或是更换模型,下面开始说明如何部署

一:训练yolo模型
1.获得数据集
从kaggle上获得有关于PCB缺陷的数据集:https://www.kaggle.com/datasets/akhatova/pcb-defects
将数据集改造成适合yolo格式的过程在此略过不提,在这里我通过数据增强的方式提升数据集的数量,加强训练的效果

2.训练自己的yolo模型
准备好配置文件,我的配置文件如下:
<img width="1027" height="401" alt="XPWA$P2GQW3L(8N)3YP3J_E" src="https://github.com/user-attachments/assets/4f63b1c9-9bba-4d3c-96af-11dd643e1bb8" />
我的显卡为rtx4070tisuper,训练的参数如batch_size等依据自己的设备来调试最佳参数,训练yolo模型的过程在此略过

3.获得训练好的模型
<img width="1047" height="560" alt="QU2`547LERQ6R$%7(XD%BPF" src="https://github.com/user-attachments/assets/ccbe2c4d-5210-4c22-84a7-20f19f9835ac" />
训练好的模型就在weight下,选取best.pt重命名为yolov8n.pt

二:连接主机和树莓派
首先要确保你的树莓派和主机在同一局域网下
烧录系统,安装树莓派等前置工作就不细讲了
1.主机使用VNC对树莓派进行连接
登录你的路由器后台,查看你对应树莓派的ip地址,将其复制下来,如:
<img width="1866" height="1209" alt="image" src="https://github.com/user-attachments/assets/61777842-44b6-4919-b41d-d07fc89e6e16" />
这里的pi1就是我对应的树莓派,名称为你自己设置的树莓派账号名称
然后下载VNC后登录树莓派:
进入树莓派桌面(如果是没有桌面版本的可以直接使用cmd如果ssh连接的形式进入),这里使用VNC时可能有一些前置措施需要做,即通过更改设置来启用VNC
<img width="1941" height="1137" alt="image" src="https://github.com/user-attachments/assets/e7cd0495-d693-489e-8865-b597c42f5404" />

注意 树莓派在关机重启后ip可能会发生变动

2.搭建树莓派相关环境
首先在home目录下创建一个新文件夹,用于存放项目文件,我将其命名为yolo
<img width="1909" height="1126" alt="image" src="https://github.com/user-attachments/assets/840a4154-3e91-44ab-8984-1338b5d1a76b" />
cd进自己对应的目录,创建虚拟环境:

python -m venv venv

然后激活

source venv/bin/activate

再输入以下指令,安装相应依赖:

pip install ultralytics ncnn -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --resume-retries 100
这里的100参数是指当你网络不好的时候可能会导致你的下载中断,它会在你中断下载的时候尝试恢复下载100回,如果你的网络情况很好可以忽略

安装好后,输入pip list查看你的环境:
<img width="790" height="801" alt="image" src="https://github.com/user-attachments/assets/1a27731c-c729-45d2-8d8d-f21a897f6dfd" />
这里依赖很多我只展示出一部分

然后我们需要对模型进行转换,再安装转换模型所需要的依赖:

pip install onnx onnxslim onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --resume-retries 100

安装完成后,我们将之前训练好的模型给导入到你的项目文件夹下,这里可以通过xftp等方式将文件输送过来
我也会在该项目中也会将代码和模型开源,直接将文件copy过去就行,篇幅原因这里不详细说明实现方式

转换模型:

yolo export model=yolov8n.pt format=ncnn
<img width="753" height="90" alt="image" src="https://github.com/user-attachments/assets/01c49f22-e7a3-4ff9-a784-65d0f7f12225" />

转换后ls查看目录,除了yolo_results其他都应该有,没有的话就重新将文件输送过来:

3.测试你的模型
首先得先找出你的摄像头,因为不同厂家,连接方式的摄像头都会导致调用方式的不同,我的摄像头型号为imx219

依据你的摄像头调用方式,更改命令,我的命令如下:

python yolo_detect.py --model=yolov8n_ncnn_model/ --source=picamera0 --resolution=640x480

出现了以下画面就成功了:
<img width="1917" height="1021" alt="image" src="https://github.com/user-attachments/assets/0edf2371-1a87-4b06-b2e4-9a154c096d01" />


三:和主机端互联
在主机端创建一个目录,将server.py放入其中(不放也行,我也会提供已经打包好能直接用的exe文件):
运行后,出现以下界面:
<img width="1422" height="955" alt="image" src="https://github.com/user-attachments/assets/a22dfcb7-e6e2-444a-8591-49574376bd4a" />

点击启动服务器,再在树莓派端输以下命令:

python yolo_detector_client.py   --model yolov8n.pt   --source picamera   --resolution 640x480   --server http://192.168.0.112:5000

<img width="1963" height="1351" alt="image" src="https://github.com/user-attachments/assets/57d23bd2-66f1-4903-a7d5-4024b060ae34" />
出现以上画面就成功了

主机端能实现直接对数据进行增删改查,实时远程监控保存数据等,记得选择文件路径,不然有些功能可能用不了,项目有些地方可能会有些bug,我没全测试出来,接下来就可以直接对其进行使用了

本项目从构思到实现为一人单独完成,可能有部分地方做的不是很好,后面可能会出一个演示视频,写了好久的文档不想写了,当把懒狗先

对你有帮助的话给个star吧,给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star给个star

<img width="619" height="459" alt="C7399065{ 31W}I{ES 1~I5" src="https://github.com/user-attachments/assets/99b7a523-b16a-49ec-983b-fd224f2b7af2" />





