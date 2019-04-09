#-*- coding:utf-8 -*-

"""
created by: longyucheng
2019/4/6

简略介绍：
    采用模式：
        同步模式(Synchronous operation)，即：连续数据流模式(Data streaming)
        阻塞调用(Blocking function calls)
    格式：
        simxSynchronous(clientID,true)                      # 同步模式
        simxStartSimulation(clientID,simx_opmode_oneshot)   # 初始化
        此时仿真在等待一个触发信号，code1不会运行
        simxSynchronousTrigger(clientID)    # 触发
        code1       # 由于被触发，code1运行
        
        simxSynchronousTrigger(clientID)    # 触发
        code2       # 由于被触发，code2运行
        simxGetPingTime(clientID)           # 阻塞作用：After this call, the code2 step is finished
        
        simxSynchronousTrigger(clientID)    # 触发
        code3       # 由于被触发，code3运行

"""

from __future__ import division
import numpy as np
import math
import time
import vrep
import os



# 配置参数
robot_velocity = 0.1        # 关节运行速度
robot_force = 30            # 能达到的最大力度
joint_angle = [0,0,0,0,30,0]

RAD2DEG = 180 / math.pi   # 常数，弧度转度数
tstep = 0.005             # 定义仿真步长

# 配置关节信息
jointNum = 6
baseName = 'UR5'
rgName = 'RG3'
jointName = 'UR5_joint'


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



# 设置仿真步长，为了保持API端与V-rep端相同步长
vrep.simxSetFloatingParameter(clientID, vrep.sim_floatparam_simulation_time_step, tstep, vrep.simx_opmode_oneshot)
# 打开同步模式来控制vrep
vrep.simxSynchronous(clientID, True)
# 仿真初始化
vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)



# 读取Base和Joint的句柄
jointHandle = np.zeros((jointNum, 1), dtype=np.int)
for i in range(jointNum):
    _, returnHandle = vrep.simxGetObjectHandle(clientID, jointName + str(i+1), vrep.simx_opmode_blocking)
    jointHandle[i] = returnHandle

_, baseHandle = vrep.simxGetObjectHandle(clientID, baseName, vrep.simx_opmode_blocking)
_, rgHandle = vrep.simxGetObjectHandle(clientID, rgName, vrep.simx_opmode_blocking)

print('Handles available!')
print("Handles:  ")
for i in range(len(jointHandle)):
    print("jointHandle" + str(i+1) + ": ", end = '')
    print(jointHandle[i])
print("======================")



# 首次读取每个关节配置以及末端位置，后面还会读取
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

# 获取初始仿真时间和初始配置
currCmdTime = vrep.simxGetLastCmdTime(clientID)
lastCmdTime = currCmdTime



# 开始仿真
t = 0
vrep.simxSynchronousTrigger(clientID)
while vrep.simxGetConnectionId(clientID) != -1:
    currCmdTime = vrep.simxGetLastCmdTime(clientID)
    dt = currCmdTime - lastCmdTime
    
    # 读取当前的状态配置
    for i in range(jointNum):
        _, jpos = vrep.simxGetJointPosition(clientID, jointHandle[i], vrep.simx_opmode_buffer)
        jointConfig[i] = jpos
        print(round(jpos * RAD2DEG, 2), end='  ')
    print('\n')
    
    # 暂停通信，用于存储所有控制命令一起发送
    vrep.simxPauseCommunication(clientID, True)
    for i in range(jointNum):
        vrep.simxSetJointTargetPosition(clientID, jointHandle[i], joint_angle[i]/RAD2DEG, vrep.simx_opmode_oneshot)
    vrep.simxPauseCommunication(clientID, False)
    
    res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID, rgName,\
                                                    vrep.sim_scripttype_childscript,'rg2Close',[],[],[],b'',vrep.simx_opmode_blocking)
    
    t = t + dt
    lastCmdTime = currCmdTime    # 记录当前时间
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)    # 使当前step走完,以便仿真结束后程序能停下,不然陷入死循环
    
vrep.simxFinish(clientID)
print('program ended')
print("运行时间: " + str(round(t/1000, 2)) + "(s)")



