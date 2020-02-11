#-*- coding:utf-8 -*-

"""
keyboard Instructions:
    robot moving velocity: <=5(advise)
    Q,W: joint 0
    A,S: joint 1
    Z,X: joint 2
    E,R: joint 3
    D,F: joint 4
    C,V: joint 5
    P: exit()
    T: close RG2
    Y: open RG2
    L: reset robot
    SPACE: save image
"""

import os
import cv2
import sys
import math
import time
import random
import string
import pygame
import vrep
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

    # communication and read the handles
    def __init__(self):
        jointNum = self.jointNum
        baseName = self.baseName
        rgName = self.rgName
        jointName = self.jointName
        camera_rgb_Name = self.camera_rgb_Name
        camera_depth_Name = self.camera_depth_Name
        
        print('Simulation started')
        vrep.simxFinish(-1)     # 关闭潜在的连接
        # 每隔0.2s检测一次, 直到连接上V-rep
        while True:
            # simxStart的参数分别为：服务端IP地址(连接本机用127.0.0.1);端口号;是否等待服务端开启;连接丢失时是否尝试再次连接;超时时间(ms);数据传输间隔(越小越快)
            clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
            if clientID > -1:
                print("Connection success!")
                break
            else:
                time.sleep(0.2)
                print("Failed connecting to remote API server!")
                print("Maybe you forget to run the simulation on vrep...")

        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)    # 仿真初始化

        # 读取Base和Joint的句柄
        jointHandle = np.zeros((jointNum, 1), dtype=np.int)
        for i in range(jointNum):
            _, returnHandle = vrep.simxGetObjectHandle(clientID, jointName + str(i+1), vrep.simx_opmode_blocking)
            jointHandle[i] = returnHandle

        _, baseHandle = vrep.simxGetObjectHandle(clientID, baseName, vrep.simx_opmode_blocking)
        _, rgHandle = vrep.simxGetObjectHandle(clientID, rgName, vrep.simx_opmode_blocking)
        _, cameraRGBHandle = vrep.simxGetObjectHandle(clientID, camera_rgb_Name, vrep.simx_opmode_blocking)
        _, cameraDepthHandle = vrep.simxGetObjectHandle(clientID, camera_depth_Name, vrep.simx_opmode_blocking)
        
        # 读取每个关节角度
        jointConfig = np.zeros((jointNum, 1))
        for i in range(jointNum):
             _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_blocking)
             jointConfig[i] = jpos
             
        self.clientID = clientID
        self.jointHandle = jointHandle
        self.rgHandle = rgHandle
        self.cameraRGBHandle = cameraRGBHandle
        self.cameraDepthHandle = cameraDepthHandle
        self.jointConfig = jointConfig
        
    def __del__(self):
        clientID = self.clientID
        vrep.simxFinish(clientID)
        print('Simulation end')
        
    # show Handles information
    def showHandles(self):
        
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
            print("jointHandle" + str(i+1) + ": " + jointHandle[i])
        print("rgHandle:" + rgHandle)
        print("cameraRGBHandle:" + cameraRGBHandle)
        print("cameraDepthHandle:" + cameraDepthHandle)
        print("===============================================")
        
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
        
        # 黑白取反
        height, width, channels = result_depth.shape
        for row in range(height):
            for list in range(width):
                for c in range(channels):
                    pv = result_depth[row, list, c]
                    result_depth[row, list, c] = 255 - pv
                
        return result_depth
        
    # open rg2
    def openRG2(self):
        rgName = self.rgName
        clientID = self.clientID
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, rgName,\
                                                        vrep.sim_scripttype_childscript,'rg2Open',[],[],[],b'',vrep.simx_opmode_blocking)
        
    # close rg2
    def closeRG2(self):
        rgName = self.rgName
        clientID = self.clientID
        res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, rgName,\
                                                        vrep.sim_scripttype_childscript,'rg2Close',[],[],[],b'',vrep.simx_opmode_blocking)
        
    # joint_angle是这种形式: [0,0,0,0,0,0], 所有的关节都旋转到对应的角度
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
        
    # 将第num个关节正转angle度
    def rotateCertainAnglePositive(self, num, angle):
        clientID = self.clientID
        RAD2DEG = self.RAD2DEG
        jointHandle = self.jointHandle
        jointConfig = self.jointConfig
        
        vrep.simxSetJointTargetPosition(clientID, jointHandle[num], (jointConfig[num]+angle)/RAD2DEG, vrep.simx_opmode_oneshot)
        jointConfig[num] = jointConfig[num] + angle
        
        self.jointConfig = jointConfig
        
    # 将第num个关节反转angle度
    def rotateCertainAngleNegative(self, num, angle):
        clientID = self.clientID
        RAD2DEG = self.RAD2DEG
        jointHandle = self.jointHandle
        jointConfig = self.jointConfig
        
        vrep.simxSetJointTargetPosition(clientID, jointHandle[num], (jointConfig[num]-angle)/RAD2DEG, vrep.simx_opmode_oneshot)
        jointConfig[num] = jointConfig[num] - angle
        
        self.jointConfig = jointConfig
        
    # convert array from vrep to image
    def arrayToImage(self):
        path = "imgTemp\\frame.jpg"
        if os.path.exists(path):
            os.remove(path)
        ig = self.getImageRGB()
        cv2.imwrite(path, ig)
    
    # convert array from vrep to depth image
    def arrayToDepthImage(self):
        path = "imgTempDep\\frame.jpg"
        if os.path.exists(path):
            os.remove(path)
        ig = self.getImageDepth()
        cv2.imwrite(path, ig)
    
# control robot by keyboard
def main():
    robot = UR5_RG2()
    resolutionX = robot.resolutionX
    resolutionY = robot.resolutionY
    
    #angle = float(eval(input("please input velocity: ")))
    angle = 1
    
    pygame.init()
    screen = pygame.display.set_mode((resolutionX, resolutionY))
    screen.fill((255,255,255))
    pygame.display.set_caption("Vrep yolov3 ddpg pytorch")
    # 循环事件，按住一个键可以持续移动
    pygame.key.set_repeat(200,50)
    
    while True:
        robot.arrayToImage()
        ig = pygame.image.load("imgTemp\\frame.jpg")
        #robot.arrayToDepthImage()
        #ig = pygame.image.load("imgTempDep\\frame.jpg")
        screen.blit(ig, (0, 0))
        pygame.display.update()
        
        key_pressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            # 关闭程序
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    sys.exit()
                # joinit 0
                elif event.key == pygame.K_q:
                    robot.rotateCertainAnglePositive(0, angle)
                elif event.key == pygame.K_w:
                    robot.rotateCertainAngleNegative(0, angle)
                # joinit 1
                elif event.key == pygame.K_a:
                    robot.rotateCertainAnglePositive(1, angle)
                elif event.key == pygame.K_s:
                    robot.rotateCertainAngleNegative(1, angle)
                # joinit 2
                elif event.key == pygame.K_z:
                    robot.rotateCertainAnglePositive(2, angle)
                elif event.key == pygame.K_x:
                    robot.rotateCertainAngleNegative(2, angle)
                # joinit 3
                elif event.key == pygame.K_e:
                    robot.rotateCertainAnglePositive(3, angle)
                elif event.key == pygame.K_r:
                    robot.rotateCertainAngleNegative(3, angle)
                # joinit 4
                elif event.key == pygame.K_d:
                    robot.rotateCertainAnglePositive(4, angle)
                elif event.key == pygame.K_f:
                    robot.rotateCertainAngleNegative(4, angle)
                # joinit 5
                elif event.key == pygame.K_c:
                    robot.rotateCertainAnglePositive(5, angle)
                elif event.key == pygame.K_v:
                    robot.rotateCertainAngleNegative(5, angle)
                # close RG2
                elif event.key == pygame.K_t:
                    robot.closeRG2()
                # # open RG2
                elif event.key == pygame.K_y:
                    robot.openRG2()
                # save Images
                elif event.key == pygame.K_SPACE:
                    rgbImg = robot.getImageRGB()
                    depthImg = robot.getImageDepth()
                    # 随机生成8位ascii码和数字作为文件名
                    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                    cv2.imwrite("saveImg\\rgbImg\\"+ran_str+"_rgb.jpg", rgbImg)
                    cv2.imwrite("saveImg\\depthImg\\"+ran_str+"_depth.jpg", depthImg)
                    print("save image")
                # reset angle
                elif event.key == pygame.K_l:
                    robot.rotateAllAngle([0,0,0,0,0,0])
                    angle = float(eval(input("please input velocity: ")))
                else:
                    print("Invalid input, no corresponding function for this key!")
                    
if __name__ == '__main__':
    main()
    