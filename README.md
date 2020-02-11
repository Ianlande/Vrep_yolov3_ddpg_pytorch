# Vrep YOLOV3 DDPG Pytorch
YOLOV3和DDPG在V-rep上的仿真实验
------

### 文件说明
scenes：VREP场景文件夹  
cfg：各种配置文件  
imgTemp、imgTempDep、imgTempDet、saveImg：储存各种图像的文件夹  
darknet.py、darknetUtils.py、yolo.py：与YOLOV3有关的程序  
vrep.py、vrepConst.py、remoteApi.dll：与V-rep有关的程序  
pallete：调色板文件  

### `robotControl.py`、`robotDetect.py`
使用键盘控制机械臂 : 先启动VREP仿真,再启动程序  
键盘操作方法：  
robot moving velocity: <=5(advise)  
Q,W : joint 0  
A,S : joint 1  
Z,X : joint 2  
E,R : joint 3  
D,F : joint 4  
C,V : joint 5  
P : exit()  
T : close RG2  
Y : open RG2  
L : reset robot  
SPACE : save image  
