"""
    used to gather training set:
    
"""

from __future__ import division
import numpy as np
import math
import time
import lib.vrep as vrep
import os
import cv2
import random
import string
import numpy as np

# 存图地址
save_rgb_path = "training_set\\rgb_img"
save_depth_path = "training_set\\depth_img"

# 配置参数
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
os.system('pause')

# 设置每个关节的Velocity和Force
for i in range(jointNum):
    vrep.simxSetJointTargetVelocity(clientID,jointHandle[i],robot_velocity,vrep.simx_opmode_oneshot)
    vrep.simxSetJointForce(clientID,jointHandle[i],robot_force,vrep.simx_opmode_oneshot)

# 开始仿真
start_time = vrep.simxGetLastCmdTime(clientID)

# 第一次获取rgb图像
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

cv2.namedWindow("kinect_rgb")
cv2.imshow("kinect_rgb", result_rgb)

# 第一次获取深度图像
res2, resolution2, image_depth = vrep.simxGetVisionSensorImage(clientID, cameraDepthHandle, 0, vrep.simx_opmode_blocking)

image_depth_r = [image_depth[i] for i in range(0,len(image_depth),3)]
image_depth_r = np.array(image_depth_r)
image_depth_r = image_depth_r.reshape(resolutionX,resolutionY)
image_depth_r = image_depth_r.astype(np.uint8)

image_depth_g = [image_depth[i] for i in range(1,len(image_depth),3)]
image_depth_g = np.array(image_depth_g)
image_depth_g = image_depth_g.reshape(resolutionX,resolutionY)
image_depth_g = image_depth_g.astype(np.uint8)

image_depth_b = [image_depth[i] for i in range(2,len(image_depth),3)]
image_depth_b = np.array(image_depth_b)
image_depth_b = image_depth_b.reshape(resolutionX,resolutionY)
image_depth_b = image_depth_b.astype(np.uint8)

result_depth = cv2.merge([image_depth_b,image_depth_g,image_depth_r])

cv2.namedWindow("image_depth")
cv2.imshow("image_depth", result_depth)
cv2.waitKey(0)

i = 0
while True:
    i+=1
    print("please input each angle of UR5: ")
    for each in range(jointNum):
        join = eval(input("the " + str(each+1) + "angle of UR5: "))
        joint_angle[each] = float(join)
    print("each angle of UR5: ",end='')
    
    # moving ur5
    vrep.simxPauseCommunication(clientID, True)
    for i in range(jointNum):
        vrep.simxSetJointTargetPosition(clientID, jointHandle[i], joint_angle[i]/RAD2DEG, vrep.simx_opmode_oneshot)
    vrep.simxPauseCommunication(clientID, False)
    print("please press any key when the UR5 arrive at the appointed position...")
    os.system('pause')
    
    # rgb
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
    
    cv2.imshow("kinect_rgb", result_rgb)
    cv2.waitKey(0)
    # save
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    cv2.imwrite(save_rgb_path+"\\"+ran_str+"_rgb.jpg", result_rgb)
    print("the photo saved...")
    
    
    res2, resolution2, image_depth = vrep.simxGetVisionSensorImage(clientID, cameraDepthHandle, 0, vrep.simx_opmode_blocking)
    image_depth_r = [image_depth[i] for i in range(0,len(image_depth),3)]
    image_depth_r = np.array(image_depth_r)
    image_depth_r = image_depth_r.reshape(resolutionX,resolutionY)
    image_depth_r = image_depth_r.astype(np.uint8)
    image_depth_g = [image_depth[i] for i in range(1,len(image_depth),3)]
    image_depth_g = np.array(image_depth_g)
    image_depth_g = image_depth_g.reshape(resolutionX,resolutionY)
    image_depth_g = image_depth_g.astype(np.uint8)
    image_depth_b = [image_depth[i] for i in range(2,len(image_depth),3)]
    image_depth_b = np.array(image_depth_b)
    image_depth_b = image_depth_b.reshape(resolutionX,resolutionY)
    image_depth_b = image_depth_b.astype(np.uint8)
    result_depth = cv2.merge([image_depth_b,image_depth_g,image_depth_r])
    
    cv2.imshow("kinect_rgb", result_depth)
    cv2.waitKey(0)
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    cv2.imwrite(save_depth_path+"\\"+ran_str+"_depth.jpg", result_depth)
    print("the photo saved...")
    
end_time = vrep.simxGetLastCmdTime(clientID)
t = end_time - start_time
vrep.simxFinish(clientID)
print('program ended')
print("运行时间: " + str(round(t/1000, 2)) + "(s)")


