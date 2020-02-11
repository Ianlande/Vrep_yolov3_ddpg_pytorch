from __future__ import division
import os
import cv2
import torch
import random
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
import os.path as osp
import torch.nn as nn
from darknetUtils import *
from darknet import Darknet
from torch.autograd import Variable

class YOLOV3:
    # 参数
    images = "imgTemp/frame.jpg"
    cfgfile = "cfg/yolov3-vrep-ddpg.cfg"
    namefile = "cfg/yolov3-vrep-ddpg-obj.names"
    weightsfile = "cfg/yolov3-vrep-ddpg_3000.weights"
    colorflie = "pallete"

    reso = int(416)
    num_classes = int(1)
    batch_size = int(1)
    confidence = float(0.5)
    nms_thesh = float(0.4)
    CUDA = torch.cuda.is_available()

    def __init__(self):
        namefile = self.namefile
        cfgfile = self.cfgfile
        weightsfile = self.weightsfile
        colorflie = self.colorflie
        CUDA = self.CUDA
        
        classes = load_classes(namefile)
        colors = pkl.load(open(colorflie, "rb"))
        #print(classes)

        #Set up the neural network
        print("Loading network.....")
        model = Darknet(cfgfile)
        model.load_weights(weightsfile)
        print("Network successfully loaded")

        model.net_info["height"] = 416
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32

        #If there's a GPU availible, put the model on GPU
        if CUDA:
            model.cuda()

        #Set the model in evaluation mode
        model.eval()
        
        self.inp_dim = inp_dim
        self.classes = classes
        self.colors = colors
        self.model = model
        
        print("init finish")
        
    def __del__(self):
        print("network end!")
        
    def write(self, x, results):
        colors = self.colors
        classes = self.classes
        colorflie = self.colorflie
        
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results
        cls = int(x[-1])
        #color = random.choice(colors)
        color = pkl.load(open(colorflie, "rb"))[2]
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2, color, 1)
        # 画中点
        #center = (int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2))
        # 画两点
        #cv2.circle(img, center, 2, color, -1)
        #cv2.circle(img, (320, 240), 2, color, -1)
        # 画图像中心点到目标点的直线
        #cv2.line(img, center, (320, 240), color)
        return img
        
    """
    输出的格式：(ind,x1,y1,x2,y2,s,s_cls,index_cls)
    ind是方框所属图片在这个batch中的序号
    x1,y1是在网络输入图片坐标系中，方框左上角的坐标
    x2,y2是方框右下角的坐标
    s是这个方框含有目标的得分
    s_cls是这个方框中所含目标最有可能的类别的概率得分
    index_cls是s_cls对应的这个类别在所有类别中所对应的序号
    """
    
    def detectFrame(self, frame):
        reso = self.reso
        inp_dim = self.inp_dim
        confidence = self.confidence
        num_classes = self.num_classes
        nms_thesh = self.nms_thesh
        CUDA = self.CUDA
        model = self.model
        
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)
                    
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)
        
        # type(output) != int 表示检测到了目标, 此时对目标画框后输出, 否则输出原图
        if type(output) != int:
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(int(reso)/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
            
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            list(map(lambda x: self.write(x, frame), output))
            #cv2.imshow("frame", frame)
            #cv2.waitKey(0)
            
            # coordinate
            coordinate = [int((output[0][1]+output[0][3])/2), int((output[0][2]+output[0][4])/2)]
        else:
            coordinate = -1
        return frame, coordinate
        
