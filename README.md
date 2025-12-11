# yolov8n-raspberrypi5
基于yolov8n,树莓派5的PCB缺陷检测系统


这是基于yolov8n,树莓派5的PCB缺陷检测系统,从树莓派调用摄像头,经过树莓派推理后能将推理后的图片,数据等传输至局域网下的另一台主机,从而缓解工厂流水线高强度人力劳作等场景

这是本人上传至github的第一个项目,不喜勿喷,我有玉米症,叠个甲:

本科毕设级水准!
本科毕设级水准!
本科毕设级水准!

那就是水

各个方面都有很大的提升空间,若你有更好的想法或者改进欢迎联系我,邮箱:kexuehe1i@foxmail.com,如果对你有帮助的话请点一颗stars,非常感谢!

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

2.搭建树莓派相关环境

3.寻找你的摄像头

三:在树莓派5上部署训练好的yolov8n
1.调用摄像头来进行推理

2.主机连接

3.主机端获取数据

