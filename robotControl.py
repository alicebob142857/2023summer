import sim                  
import sys
import time
import math
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import RandomSampler

class UR5_RG2:
    '''机械臂类'''
    resolutionX1 = 1920
    resolutionY1 = 1280
    resolutionX2 = 1280
    resolutionY2 = 720
    patchX = 380
    patchY = 380
    joint_angle = [0, 0, 0, 0, 0, 0]
    RAD2DEG = 180 / math.pi

    # 获取句柄
    jointNum = 6
    baseName = 'UR5'
    rgName = 'RG2'
    jointName = 'UR5_joint'
    camera_rgb_Name = './tip_rgb'
    target_Name = 'UR5L_ikTarget'
    tip_Name = 'UR5L_ikTip'
    head_Name = './head_rgb'

    def __init__(self):
        '''初始化'''
        print('Simulation started')

        # 初始化服务器客户端连接
        sim.simxFinish(-1) # 关闭之前的连接
        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        if self.clientID != -1:
            print("Connected to remote API server.")
        else:
            print("Connection not successful.")
            sys.exit("Could not connect")

        # 开始仿真
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        print("Simulation begins")

        # 获得句柄
        self.jointHandle = np.zeros((self.jointNum, 1), dtype=np.int)
        for i in range(self.jointNum):
            _, returnHandle = sim.simxGetObjectHandle(self.clientID, self.jointName + str(i+1), sim.simx_opmode_blocking)
            self.jointHandle[i] = returnHandle

        _, self.baseHandle = sim.simxGetObjectHandle(self.clientID, self.baseName, sim.simx_opmode_blocking)
        _, self.rgHandle = sim.simxGetObjectHandle(self.clientID, self.rgName, sim.simx_opmode_blocking)
        _, self.cameraRGBHandle = sim.simxGetObjectHandle(self.clientID, self.camera_rgb_Name, sim.simx_opmode_blocking)
        _, self.cameraHeadHandle = sim.simxGetObjectHandle(self.clientID, self.head_Name, sim.simx_opmode_blocking)
        _, self.targetHandle = sim.simxGetObjectHandle(self.clientID, self.target_Name, sim.simx_opmode_blocking)
        _, self.tipHandle = sim.simxGetObjectHandle(self.clientID, self.tip_Name, sim.simx_opmode_blocking)

        # 读取关节初始角度
        self.jointConfig = np.zeros((self.jointNum, 1))
        for i in range(self.jointNum):
             _, jpos = sim.simxGetJointPosition(self.clientID, self.jointHandle[i], sim.simx_opmode_blocking)
             self.jointConfig[i] = jpos

        # 初始位置
        _, self.initialPos = sim.simxGetObjectPosition(self.clientID, self.targetHandle, -1, sim.simx_opmode_blocking)
        _, self.initialOri = sim.simxGetObjectOrientation(self.clientID, self.targetHandle, -1, sim.simx_opmode_blocking)

    def __del__(self):
        '''断开与服务器的连接'''
        sim.simxFinish(self.clientID)
        print('Simulation end')

    def showHandles(self):
        '''打印句柄'''
        print('Handles available!')
        print("==============================================")
        print("Handles:  ")
        for i in range(len(self.jointHandle)):
            print("jointHandle" + str(i+1) + ": " + self.jointHandle[i])
        print("rgHandle:" + self.rgHandle)
        print("cameraRGBHandle:" + self.cameraRGBHandle)
        print("cameraDepthHandle:" + self.cameraDepthHandle)
        print("===============================================")

    def showJointAngles(self):
        '''打印关节旋转角度'''
        for i in range(self.jointNum):
            _, jpos = sim.simxGetJointPosition(self.clientID, self.jointHandle[i], sim.simx_opmode_blocking)
            print(round(float(jpos) * self.RAD2DEG, 2), end = ' ')
        print('\n')

    def getImageRGB(self):
        '''获得RGB图像'''
        res1, resolution1, image_rgb = sim.simxGetVisionSensorImage(self.clientID, self.cameraRGBHandle, 0, sim.simx_opmode_blocking)
        image_rgb = np.array(image_rgb).reshape((self.resolutionY2, self.resolutionX2, 3))
        image_rgb = np.flip(image_rgb, axis=0)
        image_rgb = image_rgb.astype(np.uint8)

        return image_rgb

    def getImageHead(self):
        '''获得RGB图像'''
        res1, resolution1, image_rgb = sim.simxGetVisionSensorImage(self.clientID, self.cameraHeadHandle, 0, sim.simx_opmode_blocking)
        image_rgb = np.array(image_rgb).reshape((self.resolutionY1, self.resolutionX1, 3))
        image_rgb = np.flip(image_rgb, axis=0)
        image_rgb = image_rgb.astype(np.uint8)

        return image_rgb

    def openRG2(self):
        '''打开机械爪'''
        res, retInts, retFloats, retStrings, retBuffer = \
            sim.simxCallScriptFunction(self.clientID, self.rgName, sim.sim_scripttype_childscript,\
                'rg2Open',[],[],[],b'',sim.simx_opmode_blocking)
        return 0

    def closeRG2(self):
        '''打开机械爪'''
        res, retInts, retFloats, retStrings, retBuffer = \
            sim.simxCallScriptFunction(self.clientID, self.rgName, sim.sim_scripttype_childscript,\
                'rg2Close',[],[],[],b'',sim.simx_opmode_blocking)
        return 0

    def rotateAllAngle(self, joint_angle):
        '''将所有关节旋转至某一角度'''
        # 暂时锁住通信，存储命令后一起发送
        sim.simxPauseCommunication(self.clientID, True)
        for i in range(self.jointNum):
            sim.simxSetJointTargetPosition(self.clientID, self.jointHandle[i], joint_angle[i]/self.RAD2DEG, sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID, False)
        
        self.jointConfig = joint_angle

    def rotateCertainAnglePositive(self, num, angle):
        '''将第num个关节正转angle度'''
        sim.simxSetJointTargetPosition(self.clientID, self.jointHandle[num], (self.jointConfig[num]+angle)/self.RAD2DEG, sim.simx_opmode_oneshot)
        self.jointConfig[num] = self.jointConfig[num] + angle
        
    def rotateCertainAngleNegative(self, num, angle):
        '''将第num个关节反转angle度'''
        sim.simxSetJointTargetPosition(self.clientID, self.jointHandle[num], (self.jointConfig[num]-angle)/self.RAD2DEG, sim.simx_opmode_oneshot)
        self.jointConfig[num] = self.jointConfig[num] - angle

    def arrayToImage(self):
        '''输出图片'''
        path = "imgTemp\\frame.jpg"
        if os.path.exists(path):
            os.remove(path)
        ig = self.getImageRGB()
        cv2.imwrite(path, ig)
    
    # 对图像进行采样的函数
    def CollectSamples(self, params, sampleNum, mode=None, prior=None):
        self.prepareGrab(params)
        originPos = self.getPosition(params)
        # 调整视角
        image = torch.tensor(self.getImageRGB()).transpose(2, 0)

        assembleList = []
        positionList = []
        resultLists = []
        # 随机采样
        if prior==None:
            crrntNum = sampleNum
            rangeY = torch.randint(int(self.patchY / 2), self.resolutionY2 - int(self.patchY / 2), (sampleNum, ))
            rangeX = torch.randint(int(self.patchX / 2), self.resolutionX2 - int(self.patchX / 2), (sampleNum, ))
            rotationList = torch.randint(0, 18, (sampleNum, ))
        # 重要性采样
        else:
            random_ratio = 0.25
            normal_ratio = 0.8
            # 正太分布的标准差
            position_sigma = 10
            angle_sigma = 1
            preNum = int(sampleNum * random_ratio)
            crrntNum = sampleNum - preNum
            normalNum = int(crrntNum * normal_ratio)
            randomNum = crrntNum - normalNum
            data = prior[0]
            rotation = prior[1]
            results = prior[2]
            position = prior[3]
            # 前一迭代的样本
            random_sampler = RandomSampler(range(sampleNum),  num_samples=preNum)
            pre_index = [ i for i in random_sampler]
            predata = data[pre_index]
            prerot = rotation[pre_index]
            preresults = results[pre_index]
            prepos = position[pre_index]

            # 以下是需要采集的样本
            # 随机获得的样本
            rangeY1 = torch.randint(int(self.patchY / 2), self.resolutionY2 - int(self.patchY / 2), (randomNum, ))
            rangeX1 = torch.randint(int(self.patchX / 2), self.resolutionX2 - int(self.patchX / 2), (randomNum, ))
            rotationRandom = torch.randint(0, 18, (randomNum, ))

            # 正态分布获取的样本
            centerPos = position[results == 1]
            centerRot = rotation[results == 1]
            centerX = centerPos[:, 0]
            centerY = centerPos[:, 1]
            rangeX2 = []
            rangeY2 = []
            rotationNormal = []
            centerNum = centerPos.shape[0]
            remainNum = normalNum
            for i in range(centerNum):
                tmpNum = remainNum / (centerNum - i)
                remainNum -= tmpNum
                normalX = torch.normal(centerX[i], position_sigma, (int(tmpNum), )).to(torch.int)
                normalY = torch.normal(centerY[i], position_sigma, (int(tmpNum), )).to(torch.int)
                print(normalX)
                normalRot = torch.normal(centerRot[i], angle_sigma, (int(tmpNum), )).to(torch.int)
                rangeX2.append(normalX)
                rangeY2.append(normalY)
                rotationNormal.append(normalRot)
            rangeX2 = torch.cat(rangeX2, dim=0)
            rangeY2 = torch.cat(rangeY2, dim=0)
            rotationNormal = torch.cat(rotationNormal, dim=0)

            rangeX = torch.cat((rangeX1, rangeX2), dim=0)
            rangeY = torch.cat((rangeY1, rangeY2), dim=0)
            rotationList = torch.cat((rotationRandom, rotationNormal), dim=0)

        for i in range(crrntNum):
            # 获得夹取的参数
            # 相机内参简化：比例系数
            ratio = 1280 / math.sqrt(0.375**2 + 0.375**2)
            nextposition = [originPos[0] + (rangeY[i] - self.resolutionY2/ 2)/ ratio, originPos[1] +\
                 (rangeX[i] - self.resolutionX2/ 2)/ ratio, 0.1]

            # position = [originPos[0], originPos[1], 0.1]
            nextrotation = [0, 0, rotationList[i] * 10 / self.RAD2DEG]
            paramTmp = nextposition + nextrotation
            positionList.append(torch.tensor(nextposition).unsqueeze(0))

            # 夹取前进行图像采集
            cropped_image = image[:, rangeX[i]-int(self.patchX/2):rangeX[i]+int(self.patchX/2), rangeY[i]-int(self.patchY/2):rangeY[i]+int(self.patchY/2)]
            resize_transform = transforms.Resize((224, 224))
            # 使用转换器对图像进行缩放
            cropped_image = resize_transform(cropped_image)
            if mode == 'show':
                path = f'learningpatch\\sample{i}.jpg'
                # print(cropped_image.shape)
                cv2.imwrite(path, np.array(cropped_image.transpose(2, 0)))
            
            # 夹取
            resultLists.append(self.grab(paramTmp))

            cropped_image = cropped_image.unsqueeze(0)
            assembleList.append(cropped_image)

            # print(paramTmp,':', results[i])

        newdata = torch.cat(assembleList, 0)
        newposition = torch.cat(positionList, 0)
        newresult = torch.tensor(resultLists)
        if prior != None:
            data = torch.cat((newdata, predata), 0)
            rotation = torch.cat((rotationList, prerot), 0)
            results = torch.cat((newresult, preresults), 0)
            position = torch.cat((newposition, prepos), 0)
        else:
            data = newdata
            rotation = rotationList
            results = newresult
            position = newposition

        return data, rotation, results, position

    def judgeGrab(self):
        '''简单的判断函数，判断是否夹取成功'''
        res, handle = sim.simxGetObjectHandle(self.clientID, 'Cuboid', sim.simx_opmode_blocking)
        res, Pos = sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_blocking)
        return int(Pos[2] > 0.2)

    def IKrouting(self, position, orientation):
        '''根据target的位置移动机械臂'''
        res = sim.simxSetObjectPosition(self.clientID, self.targetHandle, -1, position, sim.simx_opmode_blocking)
        if res != 0:
            print("Illegal Position!")
            return -1
        if orientation != None:
            # orientation = [ele/self.RAD2DEG for ele in orientation]
            res = sim.simxSetObjectOrientation(self.clientID, self.targetHandle, -1, orientation, sim.simx_opmode_blocking)
            if res != 0:
                print("Illegal Orientation!")
                return -1
        return 0

    def getPosition(self, name):
        if type(name) == str:
            res, handle = sim.simxGetObjectHandle(self.clientID, name, sim.simx_opmode_blocking)
            res, tagPos = sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_blocking)
            res, tagOri = sim.simxGetObjectOrientation(self.clientID, handle, -1, sim.simx_opmode_blocking)
            return tagPos + tagOri
        else:
            return name
    
    def move(self, name):
        num = 10
        min_height = 0.1
        if type(name) == str:
            res, handle = sim.simxGetObjectHandle(self.clientID, name, sim.simx_opmode_blocking)
            res, tagPos = sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_blocking)
            res, tagOri = sim.simxGetObjectOrientation(self.clientID, handle, -1, sim.simx_opmode_blocking)
        else:
            tagPos = name[0:3]
            tagOri = name[3:6]
        tagPos[2] = max(tagPos[2], min_height)
        res, crrntPos = sim.simxGetObjectPosition(self.clientID, self.tipHandle, -1, sim.simx_opmode_blocking)
        res, crrntOri = sim.simxGetObjectOrientation(self.clientID, self.tipHandle, -1, sim.simx_opmode_blocking)
        PosChange = np.linspace(np.array(crrntPos), np.array(tagPos), num)
        OriChange = np.linspace(np.array(crrntOri), np.array(tagOri), int(num/2))
        for i in range(num-1):
            position = list(PosChange[i+1])
            if i < num/2-1:
                orientation = list(OriChange[i+1])
            state = False
            if type(name) == str:
                res, state = sim.simxCheckCollision(self.clientID, self.rgHandle, handle, sim.simx_opmode_blocking)
            else:
                state = False
            if not state:
                self.IKrouting(position, orientation)
            else:
                if i < num/2-1:
                    orientation = list(OriChange[max(i-1, 0)])
                position = list(PosChange[max(i-1, 0)])
                self.IKrouting(position, orientation)
                break
        if type(name) == str:
            res, state = sim.simxCheckCollision(self.clientID, self.rgHandle, handle, sim.simx_opmode_streaming)
            if state:
                orientation = list(OriChange[max(num-2, 0)])
                position = list(PosChange[max(num-2, 0)])
                self.IKrouting(position, orientation)

    def initial(self):
        tmpinf = self.initialPos + self.initialOri
        self.move(tmpinf)

    def prepareGrab(self, params):
        params_local = None
        if type(params) == str:
            params_local = self.getPosition(params)
        # 0.4
        else:
            params_local = params.copy()
        params_local[2] += 0.4
        self.move(params_local)

    def grab(self, name):
        self.openRG2()
        time.sleep(1)
        self.move(name)
        self.closeRG2()
        time.sleep(1)
        self.prepareGrab(name)
        time.sleep(1)
        flag = self.judgeGrab()
        time.sleep(0.1)
        if flag == 1:
            self.move(name)
            time.sleep(1)
        self.openRG2()
        self.prepareGrab('Cuboid')
        return flag