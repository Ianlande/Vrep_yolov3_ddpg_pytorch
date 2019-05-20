#-*- coding:utf-8 -*-

"""
created by: longyucheng
2019/5/20

keyboard Instructions:
    robot moving velocity: <=5(advise)
    Q,W: joint 0
    A,S: joint 1
    Z,X: joint 2
    E,R: joint 3
    D,F: joint 4
    C,V: joint 5
    P: exit()
    T:close RG2
    Y:open RG2
    L: reset robot
    SPACE: save image

"""

import cv2
import sys
import math
import time
import random
import string
import pygame
import lib.vrep as vrep
import numpy as np

class UR5_RG2:
    # variates
    resolutionX = 640               # Camera resolution: 640*480
    resolutionY = 480
    joint_angle = [0,0,0,0,0,0]     # each angle of joint
    RAD2DEG = 180 / math.pi         # transform radian to degrees
    
    # Handles information
    jointNum = 6
    baseName = 'UR5'
    rgName = 'RG2'
    jointName = 'UR5_joint'
    camera_rgb_Name = 'kinect_rgb'
    camera_depth_Name = 'kinect_depth'

    # communication
    def __init__(self):
        jointNum = self.jointNum
        baseName = self.baseName
        rgName = self.rgName
        jointName = self.jointName
        camera_rgb_Name = self.camera_rgb_Name
        camera_depth_Name = self.camera_depth_Name
        
        print('Simulation started')
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
        _, rgHandle = vrep.simxGetObjectHandle(clientID, rgName, vrep.simx_opmode_blocking)
        _, cameraRGBHandle = vrep.simxGetObjectHandle(clientID, camera_rgb_Name, vrep.simx_opmode_blocking)
        _, cameraDepthHandle = vrep.simxGetObjectHandle(clientID, camera_depth_Name, vrep.simx_opmode_blocking)
        
        self.clientID = clientID
        self.jointHandle = jointHandle
        self.rgHandle = rgHandle
        self.cameraRGBHandle = cameraRGBHandle
        self.cameraDepthHandle = cameraDepthHandle
        
    def __del__(self):
        clientID = self.clientID
        vrep.simxFinish(clientID)
        print('Simulation end')
        
    # show Handles information
    def readHandles(self):
        
        RAD2DEG = self.RAD2DEG
        jointNum = self.jointNum
        clientID = self.clientID
        jointHandle = self.jointHandle
        rgHandle = self.rgHandle
        cameraRGBHandle = self.cameraRGBHandle
        cameraDepthHandle = self.cameraDepthHandle
        
        print('Handles available!')
        print("==============================================")
        print("Handles:  ")
        for i in range(len(jointHandle)):
            print("jointHandle" + str(i+1) + ": ", end = '')
            print(jointHandle[i])
        print("rgHandle:", end = '   ')
        print(rgHandle)
        print("cameraRGBHandle:", end = '   ')
        print(cameraRGBHandle)
        print("cameraDepthHandle:", end = '   ')
        print(cameraDepthHandle)
        print("===============================================")
        
        # 读取每个关节角度
        jointConfig = np.zeros((jointNum, 1))
        for i in range(jointNum):
             _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_blocking)
             jointConfig[i] = jpos
             
        print("===============================================")
        print("jointConfig:  ")
        for i in range(len(jointHandle)):
            print("joint angle" + str(i+1) + ": ", end = '')
            print(round(float(jointConfig[i]) * RAD2DEG, 2))
        print("===============================================")
        
        self.jointConfig = jointConfig
        
    # show each joint's angle
    def showJointAngles(self):
        RAD2DEG = self.RAD2DEG
        jointNum = self.jointNum
        clientID = self.clientID
        jointHandle = self.jointHandle
        
        for i in range(jointNum):
            _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_blocking)
            print(round(float(jpos) * RAD2DEG, 2), end = ' ')
        print('\n')
        
    # get RGB images
    def getImageRGB(self):
        clientID = self.clientID
        cameraRGBHandle = self.cameraRGBHandle
        resolutionX = self.resolutionX
        resolutionY = self.resolutionY
        
        res1, resolution1, image_rgb = vrep.simxGetVisionSensorImage(clientID, cameraRGBHandle, 0, vrep.simx_opmode_blocking)

        image_rgb_r = [image_rgb[i] for i in range(0,len(image_rgb),3)]
        image_rgb_r = np.array(image_rgb_r)
        image_rgb_r = image_rgb_r.reshape(resolutionY,resolutionX)
        image_rgb_r = image_rgb_r.astype(np.uint8)

        image_rgb_g = [image_rgb[i] for i in range(1,len(image_rgb),3)]
        image_rgb_g = np.array(image_rgb_g)
        image_rgb_g = image_rgb_g.reshape(resolutionY,resolutionX)
        image_rgb_g = image_rgb_g.astype(np.uint8)

        image_rgb_b = [image_rgb[i] for i in range(2,len(image_rgb),3)]
        image_rgb_b = np.array(image_rgb_b)
        image_rgb_b = image_rgb_b.reshape(resolutionY,resolutionX)
        image_rgb_b = image_rgb_b.astype(np.uint8)

        result_rgb = cv2.merge([image_rgb_b,image_rgb_g,image_rgb_r])
        # 镜像翻转, opencv在这里返回的是一张翻转的图
        result_rgb = cv2.flip(result_rgb, 0)
        return result_rgb
        
    # get depth images
    def getImageDepth(self):
        clientID = self.clientID
        cameraDepthHandle = self.cameraDepthHandle
        resolutionX = self.resolutionX
        resolutionY = self.resolutionY
        
        res2, resolution2, image_depth = vrep.simxGetVisionSensorImage(clientID, cameraDepthHandle, 0, vrep.simx_opmode_blocking)

        image_depth_r = [image_depth[i] for i in range(0,len(image_depth),3)]
        image_depth_r = np.array(image_depth_r)
        image_depth_r = image_depth_r.reshape(resolutionY,resolutionX)
        image_depth_r = image_depth_r.astype(np.uint8)
        
        image_depth_g = [image_depth[i] for i in range(1,len(image_depth),3)]
        image_depth_g = np.array(image_depth_g)
        image_depth_g = image_depth_g.reshape(resolutionY,resolutionX)
        image_depth_g = image_depth_g.astype(np.uint8)
        
        image_depth_b = [image_depth[i] for i in range(2,len(image_depth),3)]
        image_depth_b = np.array(image_depth_b)
        image_depth_b = image_depth_b.reshape(resolutionY,resolutionX)
        image_depth_b = image_depth_b.astype(np.uint8)
        
        result_depth = cv2.merge([image_depth_b,image_depth_g,image_depth_r])
        # 镜像翻转, opencv在这里返回的是一张翻转的图
        result_depth = cv2.flip(result_depth, 0)
        return result_depth
        
    # rotate angle
    def rotateAllAngle(self, joint_angle):
        clientID = self.clientID
        jointNum = self.jointNum
        RAD2DEG = self.RAD2DEG
        jointHandle = self.jointHandle
        
        # 暂停通信，用于存储所有控制命令一起发送
        vrep.simxPauseCommunication(clientID, True)
        for i in range(jointNum):
            vrep.simxSetJointTargetPosition(clientID, jointHandle[i], joint_angle[i]/RAD2DEG, vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(clientID, False)
        
        self.jointConfig = joint_angle
        
    # close rg2
    def closeRG2(self):
        rgName = self.rgName
        clientID = self.clientID
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, rgName,\
                                                        vrep.sim_scripttype_childscript,'rg2Close',[],[],[],b'',vrep.simx_opmode_blocking)
        
    # open rg2
    def openRG2(self):
        rgName = self.rgName
        clientID = self.clientID
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, rgName,\
                                                        vrep.sim_scripttype_childscript,'rg2Open',[],[],[],b'',vrep.simx_opmode_blocking)
        
    # subfunction, used by keyboardControl(), a joint rotate an angle positively
    def rotateEachAnglePositive(self, num, angle):
        clientID = self.clientID
        RAD2DEG = self.RAD2DEG
        jointHandle = self.jointHandle
        jointConfig = self.jointConfig
        
        vrep.simxSetJointTargetPosition(clientID, jointHandle[num], (jointConfig[num]+angle)/RAD2DEG, vrep.simx_opmode_oneshot)
        jointConfig[num] = jointConfig[num] + angle
        
        self.jointConfig = jointConfig
        
    # subfunction, used by keyboardControl(), a joint rotate an angle negatively
    def rotateEachAngleNegative(self, num, angle):
        clientID = self.clientID
        RAD2DEG = self.RAD2DEG
        jointHandle = self.jointHandle
        jointConfig = self.jointConfig
        
        vrep.simxSetJointTargetPosition(clientID, jointHandle[num], (jointConfig[num]-angle)/RAD2DEG, vrep.simx_opmode_oneshot)
        jointConfig[num] = jointConfig[num] - angle
        
        self.jointConfig = jointConfig
        
    # control robot by keyboard
    def keyboardControl(self):
        resolutionX = self.resolutionX
        resolutionY = self.resolutionY
        
        angle = float(eval(input("please input velocity: ")))
        
        pygame.init()
        screen = pygame.display.set_mode((resolutionX, resolutionY))
        screen.fill((255,255,255))
        # 循环事件，按住一个键可以持续移动
        pygame.key.set_repeat(200,50)
        
        while True:
            # 无限显示角度, 但是电脑运行速度会下降
            #self.showJointAngles()
            
            key_pressed = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                # 关闭程序
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        sys.exit()
                    # joinit 0
                    elif event.key == pygame.K_q:
                        self.rotateEachAnglePositive(0, angle)
                    elif event.key == pygame.K_w:
                        self.rotateEachAngleNegative(0, angle)
                    # joinit 1
                    elif event.key == pygame.K_a:
                        self.rotateEachAnglePositive(1, angle)
                    elif event.key == pygame.K_s:
                        self.rotateEachAngleNegative(1, angle)
                    # joinit 2
                    elif event.key == pygame.K_z:
                        self.rotateEachAnglePositive(2, angle)
                    elif event.key == pygame.K_x:
                        self.rotateEachAngleNegative(2, angle)
                    # joinit 3
                    elif event.key == pygame.K_e:
                        self.rotateEachAnglePositive(3, angle)
                    elif event.key == pygame.K_r:
                        self.rotateEachAngleNegative(3, angle)
                    # joinit 4
                    elif event.key == pygame.K_d:
                        self.rotateEachAnglePositive(4, angle)
                    elif event.key == pygame.K_f:
                        self.rotateEachAngleNegative(4, angle)
                    # joinit 5
                    elif event.key == pygame.K_c:
                        self.rotateEachAnglePositive(5, angle)
                    elif event.key == pygame.K_v:
                        self.rotateEachAngleNegative(5, angle)
                    # close RG2
                    elif event.key == pygame.K_t:
                        self.closeRG2()
                    # # open RG2
                    elif event.key == pygame.K_y:
                        self.openRG2()
                    # save Images
                    elif event.key == pygame.K_SPACE:
                        rgb_img = self.getImageRGB()
                        depth_img = self.getImageDepth()
                        # 随机生成8位ascii码和数字作为文件名
                        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                        cv2.imwrite("save_img\\rgb_img\\"+ran_str+"_rgb.jpg", rgb_img)
                        cv2.imwrite("save_img\\depth_img\\"+ran_str+"_depth.jpg", depth_img)
                        print("save image")
                    # reset angle
                    elif event.key == pygame.K_l:
                        self.rotateAllAngle([0,0,0,0,0,0])
                        angle = float(eval(input("please input velocity: ")))
                    else:
                        print("Invalid input, no corresponding function for this key!")
                        
                        
                        