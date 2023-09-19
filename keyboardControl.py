from robotControl import UR5_RG2
import sys
import math
import time
import random
import string
import os
import cv2
import pygame

# main函数，用来处理键盘输入输出
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
    print(robot.getImageRGB())
    
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
                    print('P')
                    sys.exit()
                # joinit 0
                elif event.key == pygame.K_q:
                    print('Q')
                    robot.rotateCertainAnglePositive(0, angle)
                elif event.key == pygame.K_w:
                    print('W')
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
                    print('T')
                    robot.closeRG2()
                # # open RG2
                elif event.key == pygame.K_y:
                    print("Y")
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
        robot.arrayToImage()
                    
if __name__ == '__main__':
    main()