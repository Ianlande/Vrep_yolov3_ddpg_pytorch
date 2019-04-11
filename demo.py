#-*- coding:utf-8 -*-

"""
created by: longyucheng
2019/4/6

简略介绍：
    采用模式：
        数据流模式(Data streaming)
        非阻塞式函数调用模式(Non-blocking function calls),用于set命令
        阻塞调用(Blocking function calls),用于get命令
"""

from __future__ import division
import numpy as np
import math
import time
import lib.vrep as vrep
import os


# 配置参数
robot_velocity = 0.01        # 关节运行速度
robot_force = 30            # 能达到的最大力度
joint_angle = [0,0,0,90,30,0]   #每个关节转动的角度(度数)

RAD2DEG = 180 / math.pi   # 常数，弧度转度数
tstep = 0.005             # 定义仿真步长

# Handles information
jointNum = 6
baseName = 'UR5'
rgName = 'RG2'
jointName = 'UR5_joint'
camera_rgb_Name = 'kinect_rgb'
camera_depth_Name = 'kinect_depth'

# 初始化
print('Program started')
# 关闭潜在的连接
vrep.simxFinish(-1)
# 每隔0.2s检测一次，直到连接上V-rep
while True:
    # 参数：服务端IP地址(本机为127.0.0.1)，端口号，是否等待服务端开启，连接丢失时是否尝试再次连接，超时时间(ms)，数据传输间隔(越小越快)
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if clientID > -1:
        break
    else:
        time.sleep(0.2)
        print("Failed connecting to remote API server!")
        print("Maybe you forget to run the simulation on vrep...")
print("Connection success!")

# 同步
vrep.simxSynchronous(clientID,True)
# 仿真初始化
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

# 读取Base和Joint的句柄
jointHandle = np.zeros((jointNum, 1), dtype=np.int)
for i in range(jointNum):
    _, returnHandle = vrep.simxGetObjectHandle(clientID, jointName + str(i+1), vrep.simx_opmode_blocking)
    jointHandle[i] = returnHandle

_, baseHandle = vrep.simxGetObjectHandle(clientID, baseName, vrep.simx_opmode_blocking)
_, rgHandle = vrep.simxGetObjectHandle(clientID, rgName, vrep.simx_opmode_blocking)
_, cameraRGBHandle = vrep.simxGetObjectHandle(clientID, camera_rgb_Name, vrep.simx_opmode_blocking)
_, cameraDepthHandle = vrep.simxGetObjectHandle(clientID, camera_depth_Name, vrep.simx_opmode_blocking)

print('Handles available!')
print("Handles:  ")
for i in range(len(jointHandle)):
    print("jointHandle" + str(i+1) + ": ", end = '')
    print(jointHandle[i])
print("rgHandle:")
print(rgHandle)
print("cameraRGBHandle:")
print(cameraRGBHandle)
print("cameraDepthHandle:")
print(cameraDepthHandle)
print("======================")



# 首次读取每个关节配置以及末端位置
jointConfig = np.zeros((jointNum, 1))
for i in range(jointNum):
    # 参数：simx_opmode_streaming(连续数据流模式)
     _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_streaming)
     jointConfig[i] = jpos

print("jointConfig:  ")
for i in range(len(jointHandle)):
    print("jointConfig" + str(i+1) + ": ", end = '')
    print(jointConfig[i])
print("======================")
os.system('pause')

# 设置每个关节的Velocity和Force
for i in range(jointNum):
    vrep.simxSetJointTargetVelocity(clientID,jointHandle[i],robot_velocity,vrep.simx_opmode_oneshot)
    vrep.simxSetJointForce(clientID,jointHandle[i],robot_force,vrep.simx_opmode_oneshot)



# 开始仿真
start_time = vrep.simxGetLastCmdTime(clientID)
vrep.simxSynchronousTrigger(clientID)

res1, resolution1, image_rgb = vrep.simxGetVisionSensorImage(clientID, cameraRGBHandle, 1, vrep.simx_opmode_blocking  )
res2, resolution2, image_depth = vrep.simxGetVisionSensorImage(clientID, cameraDepthHandle, 1, vrep.simx_opmode_blocking  )
print('res_rgb: ',end='')
print(res1)
print('resolution: ',end='')
print(resolution1)
print('res_depth: ',end='')
print(res2)
print('resolution: ',end='')
print(resolution2)

# moving ur5
# 暂停通信，用于存储所有控制命令一起发送
vrep.simxPauseCommunication(clientID, True)
for i in range(jointNum):
    vrep.simxSetJointTargetPosition(clientID, jointHandle[i], joint_angle[i]/RAD2DEG, vrep.simx_opmode_oneshot)
vrep.simxPauseCommunication(clientID, False)

# close rg2
res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, rgName,\
                                                vrep.sim_scripttype_childscript,'rg2Close',[],[],[],b'',vrep.simx_opmode_blocking)
                                                

while vrep.simxGetConnectionId(clientID) != -1:
    
    for i in range(jointNum):
        _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_buffer)
        jointConfig[i] = jpos
        print(round(jpos * RAD2DEG, 2), end='  ')
    print('\n')
    
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)    # 使当前step走完,以便仿真结束后程序能停下,不然陷入死循环
    
    
end_time = vrep.simxGetLastCmdTime(clientID)
t = end_time - start_time
vrep.simxFinish(clientID)
print('program ended')
print("运行时间: " + str(round(t/1000, 2)) + "(s)")



