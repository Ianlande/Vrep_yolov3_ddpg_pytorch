#-*- coding:utf-8 -*-

from robotControl import *
from yolo import *
from DDPG import *

import time
import math

# 强化学习参数
state_dim = 2 # 状态个数
action_dim = 3 # 动作个数
max_action = 5.0 # 动作最大值

# 其它参数
num_episodes = 1000     # 训练时走几次
num_steps = 10     # 训练时一次走几步
test_iteration = 1     # 测试时走几次
num_test_steps = 10    # 测试时一次走几步
mode = 'train'      # train or test

retrain = True        # 是否重头训练
weight_num = 900        # 载入权重的代数,用于中途继续训练和test情况
log_interval = 100       # 每隔log_interval保存一次参数
print_log = 5       # 每走print_log次输出一次
exploration_noise = 0.1 # 加入随机量
capacity = 5000 # 储存量

robot = UR5_RG2()
yolo = YOLOV3()
resolutionX = robot.resolutionX
resolutionY = robot.resolutionY

# create the directory to save the weight and the result
directory = './exp_ddpg./'
if not os.path.exists(directory):
    os.mkdir(directory)

# use the cuda or not
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print('using the GPU...')
else:
    print('using the CPU...')

# 创建agent
agent = DDPG(state_dim, action_dim, max_action,capacity,device)

# train
if mode == 'train':
    # 是否中途开始训练
    if retrain == False:
        agent.load(directory, weight_num)
    
    for i_episode in range(num_episodes):
        
        # 环境回归原位
        robot.rotateAllAngle([90,-45,90,0,-100,0])
        time.sleep(1)
        robot.arrayToImage()
        img = cv2.imread("imgTemp\\frame.jpg")
        
        frame, coord = yolo.detectFrame(img)
        # 如果检测到物体
        if coord != -1:
            # 获取距离中点信息
            xlen = coord[0]-320
            ylen = coord[1]-240
            len = round(math.hypot(xlen, ylen),2)
            # 获取深度信息
            robot.arrayToDepthImage()
            depthImg = cv2.imread("imgTempDet\\frame.jpg") # float两小数
            depLen = depthImg[coord[1], coord[0]][0] # vrep图像是倒转的, 所以需要坐标转一下
            state = [len, depLen] # 第一次获取state
        else:
            print("环境reset时没有检测到物体")
            break
        rewards = []
        
        # 每次走num_steps步
        for t in range(num_steps):
            done = 0
            
            # 选action
            action = agent.select_action(np.array(state))
            
            # add noise to action
            action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-5, 5)
            
            # 环境反馈
            for each in range(action_dim):
                robot.rotateCertainAnglePositive(each, action[each])
            
            # next_state
            robot.arrayToImage()
            img = cv2.imread("imgTemp\\frame.jpg")
            frame, coord = yolo.detectFrame(img)
            if coord != -1:
                # 获取距离中点信息
                xlen = coord[0]-320
                ylen = coord[1]-240
                len = round(math.hypot(xlen, ylen),2)
                # 获取深度信息
                robot.arrayToDepthImage()
                depthImg = cv2.imread("imgTempDet\\frame.jpg")
                depLen = depthImg[coord[1], coord[0]][0]
                next_state = [len, depLen]
            else :
                print("检测不到物体")
                done = 1
                break
                
            # reward
            reward = (1/(0.1*len)) + (1/(0.1*depLen))
            
            rewards.append(reward)
            agent.replay_buffer.push((np.array(state), np.array(next_state), np.array(action), reward, np.float(done)))
            
            # 更新state
            state = next_state
            
        # 参数更新是运行完一次更新一次, 不是每走一步更新一次
        #if len(agent.replay_buffer.storage) >= capacity-1:
        #    agent.update()
        agent.update()
            
        # 保存权重并输出
        if i_episode % log_interval == 0 and i_episode != 0:
            agent.save(directory, i_episode)
            
        print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
# test
elif mode == 'test':
    agent.load(directory, weight_num)
    print("load weight...")
    
    for i_episode in range(test_iteration):
        state = env.reset()
        for t in range(num_test_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(np.float32(action))
            env.render()
            state = next_state
            if done:
                break
else:
    raise NameError("mode wrong!!!")


