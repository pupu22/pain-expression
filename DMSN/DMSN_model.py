import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def conv_T(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1), bias=False)


def conv_S(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), bias=False)


'''
    DMSN中Bottleneck表示DMSN变体的基本结构
    Block的各个plane值：
        inplane：输入block的之前的通道数
        midplane：在block中间处理的时候的通道数（这个值是输出维度的1/4）
        midplane*self.extention：输出的维度
'''


class ModuleA(nn.Module):
    extention = 4

    def __init__(self, inplane, midplane, stride, st_struc='A'):
        super(ModuleA, self).__init__()
        self.st_struc = st_struc
        self.midplane = midplane

        print(self.midplane)

        # stage1输入通道数为64，main stage中通道数为他的一半，分支通道数为他的四分之一
        # 这里stage1输入了16，first_plane为32，double_plane为64

        one_plane = midplane * 2
        double_plane = midplane * self.extention

        self.conv1 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=(1, 1, 1),
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(midplane * self.extention)
        self.conv2 = conv_T(double_plane, one_plane)
        self.bn2 = nn.BatchNorm3d(one_plane)
        self.conv6 = conv_S(one_plane, midplane)
        self.bn6 = nn.BatchNorm3d(midplane)

        self.conv3 = conv_T(one_plane, one_plane)
        self.bn3 = nn.BatchNorm3d(one_plane)
        self.conv7 = conv_S(one_plane, midplane)
        self.bn7 = nn.BatchNorm3d(midplane)

        self.conv4 = conv_T(one_plane, one_plane)
        self.bn4 = nn.BatchNorm3d(one_plane)
        self.conv8 = conv_S(one_plane, midplane)
        self.bn8 = nn.BatchNorm3d(midplane)

        self.conv5 = conv_T(one_plane, one_plane)
        self.bn5 = nn.BatchNorm3d(one_plane)
        self.conv9 = conv_S(one_plane, midplane)
        self.bn9 = nn.BatchNorm3d(midplane)

        self.conv10 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=1, stride=1,
                                bias=False)
        self.bn10 = nn.BatchNorm3d(midplane * self.extention)
        self.relu = nn.ReLU(inplace=False)

    def ST_A(self, T):
        T1 = self.conv2(T)
        T1 = self.bn2(T1)
        T1 = self.relu(T1)
        ST1 = self.conv6(T1)
        ST1 = self.bn6(ST1)
        ST1 = self.relu(ST1)
        # ST1 = np.concatenate(())

        T2 = self.conv3(T1)
        T2 = self.bn3(T2)
        T2 = self.relu(T2)
        # print(T2.size())
        ST2 = self.conv7(T2)
        ST2 = self.bn7(ST2)
        # print(ST2.size())
        ST2 = self.relu(ST2)

        T3 = self.conv4(T2)
        T3 = self.bn4(T3)
        T3 = self.relu(T3)
        # print(T3.size())
        ST3 = self.conv8(T3)
        ST3 = self.bn8(ST3)
        ST3 = self.relu(ST3)
        # print(ST3.size())
        # ST3 = np.concatenate(ST2, ST3)

        T4 = self.conv5(T3)
        T4 = self.bn5(T4)
        T4 = self.relu(T4)
        ST4 = self.conv9(T4)
        ST4 = self.bn9(ST4)
        ST4 = self.relu(ST4)
        # ST4 = np.concatenate(ST3, ST4)
        # print(ST3.size())
        # print(ST4.size())
        ST4 = np.concatenate((ST1.cpu().detach(), ST2.cpu().detach(), ST3.cpu().detach(), ST4.cpu().detach()), axis=1)
        ST4 = torch.from_numpy(ST4)

        return ST4

    def forward(self, xx):
        # 参差数据
        residual = xx

        out = self.relu(self.bn1(self.conv1(xx)))
        out = self.ST_A(out)
        out = self.relu(self.bn10(self.conv10(out)))
        out = np.concatenate((out.detach(), residual.detach()), axis=1)
        out = torch.from_numpy(out)
        out = self.relu(out)

        return out


class ModuleB(nn.Module):
    extention = 4

    def __init__(self, inplane, midplane, stride, st_struc='A'):
        super(ModuleB, self).__init__()
        self.st_struc = st_struc
        self.midplane = midplane

        one_plane = midplane * 2
        double_plane = midplane * self.extention
        self.conv1 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=(1, 1, 1),
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(midplane * self.extention)
        self.conv6 = conv_S(double_plane, one_plane)
        self.bn6 = nn.BatchNorm3d(one_plane)
        self.conv2 = conv_T(one_plane, midplane)
        self.bn2 = nn.BatchNorm3d(midplane)

        self.conv3 = conv_T(one_plane, one_plane)
        self.bn3 = nn.BatchNorm3d(one_plane)
        self.conv7 = conv_S(one_plane, midplane)
        self.bn7 = nn.BatchNorm3d(midplane)

        self.conv8 = conv_S(one_plane, one_plane)
        self.bn8 = nn.BatchNorm3d(one_plane)
        self.conv4 = conv_T(one_plane, midplane)
        self.bn4 = nn.BatchNorm3d(midplane)

        self.conv5 = conv_T(one_plane, one_plane)
        self.bn5 = nn.BatchNorm3d(one_plane)
        self.conv9 = conv_S(one_plane, midplane)
        self.bn9 = nn.BatchNorm3d(midplane)
        self.conv10 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=1, stride=1,
                                bias=False)
        self.bn10 = nn.BatchNorm3d(midplane * self.extention)
        self.relu = nn.ReLU(inplace=False)

    def ST_B(self, Y):
        Y1 = self.conv6(Y)
        Y1 = self.bn6(Y1)
        Y1 = self.relu(Y1)
        ST1 = self.conv2(Y1)
        ST1 = self.bn2(ST1)
        ST1 = self.relu(ST1)

        Y2 = self.conv3(Y1)
        Y2 = self.bn3(Y2)
        Y2 = self.relu(Y2)
        ST2 = self.conv7(Y2)
        ST2 = self.bn7(ST2)
        ST2 = self.relu(ST2)

        Y3 = self.conv8(Y2)
        Y3 = self.bn8(Y3)
        Y3 = self.relu(Y3)
        ST3 = self.conv4(Y3)
        ST3 = self.bn4(ST3)
        ST3 = self.relu(ST3)

        Y4 = self.conv5(Y3)
        Y4 = self.bn5(Y4)
        Y4 = self.relu(Y4)
        ST4 = self.conv9(Y4)
        ST4 = self.bn9(ST4)
        ST4 = self.relu(ST4)

        ST4 = np.concatenate((ST1.cpu().detach(), ST2.cpu().detach(), ST3.cpu().detach(), ST4.cpu().detach()), axis=1)
        ST4 = torch.from_numpy(ST4)
        return ST4

    def forward(self, xx):
        # 参差数据
        residual = xx

        out = self.relu(self.bn1(self.conv1(xx)))
        out = self.ST_B(out)
        out = self.relu(self.bn10(self.conv10(out)))
        out = np.concatenate((out.detach(), residual.detach()), axis=1)
        out = torch.from_numpy(out)
        out = self.relu(out)

        return out


class ModuleC(nn.Module):
    extention = 4

    def __init__(self, inplane, midplane, stride, st_struc='A'):
        super(ModuleC, self).__init__()
        self.st_struc = st_struc
        self.midplane = midplane

        one_plane = midplane * 2
        double_plane = midplane * self.extention
        self.conv1 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=(1, 1, 1),
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(midplane * self.extention)
        self.conv6 = conv_S(double_plane, one_plane)
        self.bn6 = nn.BatchNorm3d(one_plane)
        self.conv2 = conv_T(one_plane, midplane)
        self.bn2 = nn.BatchNorm3d(midplane)

        self.conv7 = conv_S(one_plane, one_plane)
        self.bn7 = nn.BatchNorm3d(one_plane)
        self.conv3 = conv_T(one_plane, midplane)
        self.bn3 = nn.BatchNorm3d(midplane)

        self.conv8 = conv_S(one_plane, one_plane)
        self.bn8 = nn.BatchNorm3d(one_plane)
        self.conv4 = conv_T(one_plane, midplane)
        self.bn4 = nn.BatchNorm3d(midplane)

        self.conv9 = conv_S(one_plane, one_plane)
        self.bn9 = nn.BatchNorm3d(one_plane)
        self.conv5 = conv_T(one_plane, midplane)
        self.bn5 = nn.BatchNorm3d(midplane)

        self.conv10 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=1, stride=1,
                                bias=False)
        self.bn10 = nn.BatchNorm3d(midplane * self.extention)
        self.relu = nn.ReLU(inplace=False)

    def ST_C(self, S):
        S1 = self.conv6(S)
        S1 = self.bn6(S1)
        S1 = self.relu(S1)
        ST1 = self.conv2(S1)
        ST1 = self.bn2(ST1)
        ST1 = self.relu(ST1)

        S2 = self.conv7(S1)
        S2 = self.bn7(S2)
        S2 = self.relu(S2)
        ST2 = self.conv3(S2)
        ST2 = self.bn3(ST2)
        ST2 = self.relu(ST2)

        S3 = self.conv8(S2)
        S3 = self.bn8(S3)
        S3 = self.relu(S3)
        ST3 = self.conv4(S3)
        ST3 = self.bn4(ST3)
        ST3 = self.relu(ST3)

        S4 = self.conv9(S3)
        S4 = self.bn9(S4)
        S4 = self.relu(S4)
        ST4 = self.conv5(S4)
        ST4 = self.bn5(ST4)
        ST4 = self.relu(ST4)

        ST4 = np.concatenate((ST1.cpu().detach(), ST2.cpu().detach(), ST3.cpu().detach(), ST4.cpu().detach()), axis=1)
        ST4 = torch.from_numpy(ST4)
        return ST4

    def forward(self, xx):
        # 参差数据
        residual = xx

        out = self.relu(self.bn1(self.conv1(xx)))
        out = self.ST_B(out)
        out = self.relu(self.bn10(self.conv10(out)))
        out = np.concatenate((out.detach(), residual.detach()), axis=1)
        out = torch.from_numpy(out)
        out = self.relu(out)

        return out


class DMSN(nn.Module):

    # 初始化网络结构和参数
    def __init__(self, layers, num_classes=1000):
        # self.inplane为当前的fm的通道数
        self.inplane = 64

        super(DMSN, self).__init__()

        # 参数
        # layers里面的值表示里面block要循环的次数
        self.layers = layers

        # stem的网络层
        self.conv = nn.Conv3d(3, self.inplane, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), padding=1, stride=2)
        # ----------------------stage1--------------------
        self.module1A = ModuleA(self.inplane, 16, stride=1, st_struc='A')
        self.module1B = ModuleB(self.inplane, 16, stride=1, st_struc='B')
        self.module1C = ModuleB(self.inplane, 16, stride=1, st_struc='C')
        # ----------------------stage2--------------------
        self.module2A = ModuleA(128, 32, stride=1, st_struc='A')
        self.module2B = ModuleB(128, 32, stride=1, st_struc='B')
        self.module2C = ModuleC(128, 32, stride=1, st_struc='C')
        self.module2AA = ModuleA(128, 32, stride=1, st_struc='A')
        # ----------------------stage3--------------------
        self.module3A = ModuleA(256, 64, stride=1, st_struc='A')
        self.module3B = ModuleB(256, 64, stride=1, st_struc='B')
        self.module3C = ModuleC(256, 64, stride=1, st_struc='C')
        self.module3AA = ModuleA(256, 64, stride=1, st_struc='A')
        self.module3BB = ModuleB(256, 64, stride=1, st_struc='B')
        self.module3CC = ModuleC(256, 64, stride=1, st_struc='C')
        # ----------------------stage4--------------------
        self.module4A = ModuleA(512, 128, stride=1, st_struc='A')
        self.module4B = ModuleB(512, 128, stride=1, st_struc='B')
        self.module4C = ModuleC(512, 128, stride=1, st_struc='C')
        self.module4AA = ModuleA(512, 128, stride=1, st_struc='A')

        # 后续的网络
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):

        # stem部分:conv+bn+relu+maxpool
        print(x.size())
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.size())
        out = self.maxpool(out)

        # block
        # -----------STAGE1------------
        # ------------A----------------
        # residual = out
        # out = nn.Conv3d(64, 64, kernel_size=(1, 1, 1), stride=1, bias=False)(out)
        # out = nn.BatchNorm3d(64)(out)
        # out = conv_T(64, 32)(out)
        # out = nn.BatchNorm3d(32)(out)
        # ST1 = conv_S(32, 16)(out)
        # ST1 = nn.BatchNorm3d(ST1)
        #
        # out = conv_T(32, 32)(out)
        # out = nn.BatchNorm3d(32)(out)
        # out = conv_S(32, 16)(out)
        # out = nn.BatchNorm3d(16)(out)
        #
        # out = conv_T(32, 32)(out)
        # out = nn.BatchNorm3d(16)(out)
        # out = conv_S(32, 32)(out)
        # out = nn.BatchNorm3d(16)(out)
        #
        # out = conv_T(32, 32)(out)
        # out = nn.BatchNorm3d(16)(out)
        # out = conv_S(32, 32)(out)
        # out = nn.BatchNorm3d(16)(out)
        #
        # out = nn.Conv3d(64, 64, kernel_size=1, stride=1, bias=False)(out)
        # out = nn.BatchNorm3d(64)(out)
        # out = nn.ReLU(inplace=False)(out)
        #
        # out = np.concatenate((out, residual), axis=1)
        # out = nn.ReLU(inplace=False)(out)
        # # ------------B----------------
        # residual = out
        # out = nn.Conv3d(128, 128, kernel_size=(1, 1, 1), stride=1, bias=False)(out)
        # self.bn1 = nn.BatchNorm3d(128)(out)
        # self.conv6 = conv_S(128, 64)(out)
        # self.bn6 = nn.BatchNorm3d(64)(out)
        # self.conv2 = conv_T(64, 32)(out)
        # self.bn2 = nn.BatchNorm3d(32)(out)
        #
        # self.conv3 = conv_T(one_plane, one_plane)(out)
        # self.bn3 = nn.BatchNorm3d(one_plane)(out)
        # self.conv7 = conv_S(one_plane, midplane)(out)
        # self.bn7 = nn.BatchNorm3d(midplane)(out)
        #
        # self.conv8 = conv_S(one_plane, one_plane)(out)
        # self.bn8 = nn.BatchNorm3d(one_plane)(out)
        # self.conv4 = conv_T(one_plane, midplane)(out)
        # self.bn4 = nn.BatchNorm3d(midplane)(out)
        #
        # self.conv5 = conv_T(one_plane, one_plane)(out)
        # self.bn5 = nn.BatchNorm3d(one_plane)(out)
        # self.conv9 = conv_S(one_plane, midplane)(out)
        # self.bn9 = nn.BatchNorm3d(midplane)(out)
        # self.conv10 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=1, stride=1,bias=False)(out)
        # self.bn10 = nn.BatchNorm3d(midplane * self.extention)(out)
        # self.relu = nn.ReLU(inplace=False)(out)
        #
        out = self.module1A(out)
        out = self.module1B(out)
        out = self.module1C(out)

        out = self.module2A(out)
        out = self.module2B(out)
        out = self.module2C(out)
        out = self.module2AA(out)

        out = self.module3A(out)
        out = self.module3B(out)
        out = self.module3C(out)
        out = self.module3AA(out)
        out = self.module3BB(out)
        out = self.module3CC(out)

        out = self.module4A(out)
        out = self.module4B(out)
        out = self.module4C(out)
        out = self.module4AA(out)

        # 分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


resnet = DMSN( [3, 4, 6, 3])
# resnet = resnet.to(device=6)
# data = torch.autograd.Variable(
#     torch.rand(8, 3, 16, 112, 112)).to(device=6)  # if modality=='Flow', please change the 2nd dimension 3==>2
# out = resnet(data)
# print(out.size(), out)
# 向网络输入一个1，3，224，224的tensor
x = torch.randn(8, 3, 16, 112, 112)
x = resnet(x)
print(resnet)
