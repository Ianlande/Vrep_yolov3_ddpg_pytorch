from robotControl import *
from yolo import *

def main():
    robot = UR5_RG2()
    yolo = YOLOV3()
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
        img = cv2.imread("imgTemp\\frame.jpg")          # 获取图片
        frame, coordinate = yolo.detectFrame(img)       # 检测
        cv2.imwrite("imgTempDet\\frame.jpg", np.array(frame))    # 储存检测结果图
        ig = pygame.image.load("imgTempDet\\frame.jpg")          # 读取检测结果图
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
    