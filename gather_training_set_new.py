"""
k: advise<=5
q,w: joint 0
a,s: joint 1
z,x: joint 2
e,r: joint 3
d,f: joint 4
c,v: joint 5
p: exit()
space: save image
L: 重置机械臂关节

"""
from __future__ import division
import pygame
import sys
import numpy as np
import math
import time
import lib.vrep as vrep
import os
import cv2
import random
import string
import numpy as np

pygame.init()

# 存图地址
save_rgb_path = "training_set\\rgb_img"

# 配置参数
angle = float(eval(input("please input k: ")))
robot_velocity = 0.001        # 关节运行速度
robot_force = 30            # 能达到的最大力度
resolutionX = 480           # 摄像机图片分辨率X: 640*480
resolutionY = 640           # 摄像机图片分辨率Y
joint_angle = [0,0,0,0,0,0]   #每个关节转动的角度(度数)
RAD2DEG = 180 / math.pi   # 常数，弧度转度数

# Handles information
jointNum = 6
baseName = 'UR5'
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

# 仿真初始化
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

# 读取Base和Joint的句柄
jointHandle = np.zeros((jointNum, 1), dtype=np.int)
for i in range(jointNum):
    _, returnHandle = vrep.simxGetObjectHandle(clientID, jointName + str(i+1), vrep.simx_opmode_blocking)
    jointHandle[i] = returnHandle

_, baseHandle = vrep.simxGetObjectHandle(clientID, baseName, vrep.simx_opmode_blocking)
_, cameraRGBHandle = vrep.simxGetObjectHandle(clientID, camera_rgb_Name, vrep.simx_opmode_blocking)
_, cameraDepthHandle = vrep.simxGetObjectHandle(clientID, camera_depth_Name, vrep.simx_opmode_blocking)

print('Handles available!')
print("Handles:  ")
for i in range(len(jointHandle)):
    print("jointHandle" + str(i+1) + ": ", end = '')
    print(jointHandle[i])
print("cameraRGBHandle:")
print(cameraRGBHandle)
print("cameraDepthHandle:")
print(cameraDepthHandle)
print("======================")


# 首次读取每个关节角度
jointConfig = np.zeros((jointNum, 1))
for i in range(jointNum):
     _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_blocking)
     jointConfig[i] = jpos

print("jointConfig:  ")
for i in range(len(jointHandle)):
    print("joint angle" + str(i+1) + ": ", end = '')
    print(round(jpos * RAD2DEG, 2))
print("======================")

# 设置每个关节的Velocity和Force
for i in range(jointNum):
    vrep.simxSetJointTargetVelocity(clientID,jointHandle[i],robot_velocity,vrep.simx_opmode_oneshot)
    vrep.simxSetJointForce(clientID,jointHandle[i],robot_force,vrep.simx_opmode_oneshot)

# 开始仿真
screen = pygame.display.set_mode((300,300))
screen.fill((255,255,255))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                sys.exit()
            # joinit 0
            elif event.key == pygame.K_q:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[0], (jointConfig[0]+angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[0] = jointConfig[0] + angle
            elif event.key == pygame.K_w:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[0], (jointConfig[0]-angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[0] = jointConfig[0] - angle
            # joinit 1
            elif event.key == pygame.K_a:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[1], (jointConfig[1]+angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[1] = jointConfig[1] + angle
            elif event.key == pygame.K_s:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[1], (jointConfig[1]-angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[1] = jointConfig[1] - angle
            # joinit 2
            elif event.key == pygame.K_z:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[2], (jointConfig[2]+angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[2] = jointConfig[2] + angle
            elif event.key == pygame.K_x:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[2], (jointConfig[2]-angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[2] = jointConfig[2] - angle
            # joinit 3
            elif event.key == pygame.K_e:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[3], (jointConfig[3]+angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[3] = jointConfig[3] + angle
            elif event.key == pygame.K_r:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[3], (jointConfig[3]-angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[3] = jointConfig[3] - angle
            # joinit 4
            elif event.key == pygame.K_d:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[4], (jointConfig[4]+angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[4] = jointConfig[4] + angle
            elif event.key == pygame.K_f:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[4], (jointConfig[4]-angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[4] = jointConfig[4] - angle
            # joinit 5
            elif event.key == pygame.K_c:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[5], (jointConfig[5]+angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[5] = jointConfig[5] + angle
            elif event.key == pygame.K_v:
                vrep.simxSetJointTargetPosition(clientID, jointHandle[5], (jointConfig[5]-angle)/RAD2DEG, vrep.simx_opmode_oneshot)
                jointConfig[5] = jointConfig[5] - angle
            # 重置
            elif event.key == pygame.K_l:
                vrep.simxPauseCommunication(clientID, True)
                for i in range(jointNum):
                    vrep.simxSetJointTargetPosition(clientID, jointHandle[i], 0/RAD2DEG, vrep.simx_opmode_oneshot)
                vrep.simxPauseCommunication(clientID, False)
                angle = float(eval(input("please input k: ")))
            # save image
            elif event.key == pygame.K_SPACE:
                res1, resolution1, image_rgb = vrep.simxGetVisionSensorImage(clientID, cameraRGBHandle, 0, vrep.simx_opmode_blocking)
                
                image_rgb_r = [image_rgb[i] for i in range(0,len(image_rgb),3)]
                image_rgb_r = np.array(image_rgb_r)
                image_rgb_r = image_rgb_r.reshape(resolutionX,resolutionY)
                image_rgb_r = image_rgb_r.astype(np.uint8)
                
                image_rgb_g = [image_rgb[i] for i in range(1,len(image_rgb),3)]
                image_rgb_g = np.array(image_rgb_g)
                image_rgb_g = image_rgb_g.reshape(resolutionX,resolutionY)
                image_rgb_g = image_rgb_g.astype(np.uint8)
                
                image_rgb_b = [image_rgb[i] for i in range(2,len(image_rgb),3)]
                image_rgb_b = np.array(image_rgb_b)
                image_rgb_b = image_rgb_b.reshape(resolutionX,resolutionY)
                image_rgb_b = image_rgb_b.astype(np.uint8)
                
                result_rgb = cv2.merge([image_rgb_b,image_rgb_g,image_rgb_r])
                # 镜像翻转
                result_rgb = cv2.flip(result_rgb, 0)
                # save
                ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                cv2.imwrite(save_rgb_path+"\\"+ran_str+"_rgb.jpg", result_rgb)
                print("the photo saved...")
            else:
                print("无效的输入,该按键没有对应功能!")

vrep.simxFinish(clientID)
