import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def conv_T(in_planes, out_planes, stride=(1, 1, 1), padding=(0, 0, 1)):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, padding=padding, bias=False)


def conv_S(in_planes, out_planes, stride=(1, 1, 1), padding=(1, 1, 0)):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 1), stride=stride, padding=padding, bias=False)


'''
    DMSN中Bottleneck表示DMSN变体的基本结构
    Block的各个plane值：
        inplane：输入block的之前的通道数
        midplane：在block中间处理的时候的通道数（这个值是输出维度的1/4）
        midplane*self.extention：输出的维度
'''


class Bottleneck(nn.Module):
    # 每个stage中维度拓展的倍数
    extention = 4

    # 定义初始化的网络和参数
    def __init__(self, inplane, midplane, StrideS, PaddingS, StrideT, PaddingT, downsample=None, st_struc='A'):
        super(Bottleneck, self).__init__()
        self.st_struc = st_struc
        self.midplane = midplane

        print(self.midplane)

        # stage1输入通道数为64，main stage中通道数为他的一半，分支通道数为他的四分之一
        # 这里stage1输入了16，first_plane为32，double_plane为64
        one_plane = midplane * 2
        double_plane = midplane * self.extention

        if self.st_struc == 'A':
            self.conv1 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=(1, 1, 1),
                                   stride=1, bias=False)
            self.bn1 = nn.BatchNorm3d(midplane * self.extention)
            self.conv2 = conv_T(double_plane, one_plane, stride=StrideT, padding=PaddingT)
            self.bn2 = nn.BatchNorm3d(one_plane)
            self.conv6 = conv_S(one_plane, midplane, stride=StrideS, padding=PaddingS)
            self.bn6 = nn.BatchNorm3d(midplane)

            self.conv3 = conv_T(one_plane, one_plane, stride=StrideT, padding=PaddingT)
            self.bn3 = nn.BatchNorm3d(one_plane)
            self.conv7 = conv_S(one_plane, midplane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn7 = nn.BatchNorm3d(midplane)

            self.conv4 = conv_T(one_plane, one_plane, stride=StrideT, padding=PaddingT)
            self.bn4 = nn.BatchNorm3d(one_plane)
            self.conv8 = conv_S(one_plane, midplane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn8 = nn.BatchNorm3d(midplane)

            self.conv5 = conv_T(one_plane, one_plane, stride=StrideT, padding=PaddingT)
            self.bn5 = nn.BatchNorm3d(one_plane)
            self.conv9 = conv_S(one_plane, midplane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn9 = nn.BatchNorm3d(midplane)

        elif self.st_struc == 'B':
            self.conv1 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=(1, 1, 1),
                                   stride=1, bias=False)
            self.bn1 = nn.BatchNorm3d(midplane * self.extention)
            self.conv6 = conv_S(double_plane, one_plane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn6 = nn.BatchNorm3d(one_plane)
            self.conv2 = conv_T(one_plane, midplane, stride=StrideT, padding=PaddingT)
            self.bn2 = nn.BatchNorm3d(midplane)

            self.conv3 = conv_T(one_plane, one_plane, stride=StrideT, padding=PaddingT)
            self.bn3 = nn.BatchNorm3d(one_plane)
            self.conv7 = conv_S(one_plane, midplane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn7 = nn.BatchNorm3d(midplane)

            self.conv8 = conv_S(one_plane, one_plane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn8 = nn.BatchNorm3d(one_plane)
            self.conv4 = conv_T(one_plane, midplane, stride=StrideT, padding=PaddingT)
            self.bn4 = nn.BatchNorm3d(midplane)

            self.conv5 = conv_T(one_plane, one_plane, stride=StrideT, padding=PaddingT)
            self.bn5 = nn.BatchNorm3d(one_plane)
            self.conv9 = conv_S(one_plane, midplane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn9 = nn.BatchNorm3d(midplane)

        elif self.st_struc == 'C':
            self.conv1 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=(1, 1, 1),
                                   stride=1, bias=False)
            self.bn1 = nn.BatchNorm3d(midplane * self.extention)
            self.conv6 = conv_S(double_plane, one_plane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn6 = nn.BatchNorm3d(one_plane)
            self.conv2 = conv_T(one_plane, midplane, stride=StrideT, padding=PaddingT)
            self.bn2 = nn.BatchNorm3d(midplane)

            self.conv7 = conv_S(one_plane, one_plane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn7 = nn.BatchNorm3d(one_plane)
            self.conv3 = conv_T(one_plane, midplane, stride=StrideT, padding=PaddingT)
            self.bn3 = nn.BatchNorm3d(midplane)

            self.conv8 = conv_S(one_plane, one_plane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn8 = nn.BatchNorm3d(one_plane)
            self.conv4 = conv_T(one_plane, midplane, stride=StrideT, padding=PaddingT)
            self.bn4 = nn.BatchNorm3d(midplane)

            self.conv9 = conv_S(one_plane, one_plane, stride=(1, 1, 1), padding=(1, 1, 0))
            self.bn9 = nn.BatchNorm3d(one_plane)
            self.conv5 = conv_T(one_plane, midplane, stride=StrideT, padding=PaddingT)
            self.bn5 = nn.BatchNorm3d(midplane)

        self.conv10 = nn.Conv3d(midplane*self.extention, midplane*self.extention, kernel_size=1, stride=1, bias=False)
        self.bn10 = nn.BatchNorm3d(midplane * self.extention)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = 1

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

        if self.st_struc == 'A':
            out = self.ST_A(out)
        elif self.st_struc == 'B':
            out = self.ST_B(out)
        elif self.st_struc == 'C':
            out = self.ST_C(out)

        out = self.relu(self.bn10(self.conv10(out)))
        # out = np.concatenate((out.detach(), residual.detach()), axis=1)
        # out = torch.from_numpy(out)

        out += residual
        out = self.relu(out)
        print(out.size())
        return out


class DMSN(nn.Module):

    # 初始化网络结构和参数
    def __init__(self, block, layers, num_classes=1000):
        # self.inplane为当前的fm的通道数
        self.inplane = 64

        super(DMSN, self).__init__()

        # 参数
        # layers里面的值表示里面block要循环的次数
        self.block = block
        self.layers = layers

        # stem的网络层
        self.conv = nn.Conv3d(3, self.inplane, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), padding=1, stride=2)

        # 32，64，128，256是指扩大4倍之前的维度,即Identity Block的中间维度
        self.ConvSize = nn.Conv3d(3, self.inplane, kernel_size=(1, 1, 1), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.stage1 = self.make_layer(self.block, 16, self.layers[0], shortcut_type=1, stride=1)
        self.stage2 = self.make_layer(self.block, 32, self.layers[1], shortcut_type=2, stride=2)
        self.stage3 = self.make_layer(self.block, 64, self.layers[2], shortcut_type=3, stride=2)
        self.stage4 = self.make_layer(self.block, 128, self.layers[3], shortcut_type=4, stride=2)

        # 后续的网络
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.extention, num_classes)

    def forward(self, x):

        # stem部分:conv+bn+relu+maxpool
        print(x.size())
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.size())
        out = self.maxpool(out)

        # block
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # 分类
        # out = self.avgpool(out)
        # out = torch.flatten(out, 1)
        # out = self.fc(out)

        return out

    def make_layer(self, block, midplane, block_num, shortcut_type, stride=1):
        '''
            block:block模块
            midplane：每个模块中间运算的维度，一般等于输出维度/4
            block_num：重复次数
            stride：Conv Block的步长
        '''

        block_list = []
        if shortcut_type == 1:
            block_list.append(block(self.inplane, 16, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(block(self.inplane, 16, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='B'))
            block_list.append(block(self.inplane, 16, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='C'))
            block_list.append(nn.Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                        padding=(0, 0, 0), bias=False))
        elif shortcut_type == 2:
            block_list.append(block(128, 32, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(block(128, 32, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='B'))
            block_list.append(block(128, 32, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='C'))
            block_list.append(block(128, 32, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(nn.Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                        padding=(0, 0, 0), bias=False))
        elif shortcut_type == 3:
            block_list.append(block(256, 64, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(block(256, 64, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='B'))
            block_list.append(block(256, 64, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='C'))
            block_list.append(block(256, 64, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(block(256, 64, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='B'))
            block_list.append(block(256, 64, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='C'))
            block_list.append(nn.Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                        padding=(0, 0, 0), bias=False))
        elif shortcut_type == 4:
            block_list.append(block(512, 128, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(block(512, 128, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='B'))
            block_list.append(block(512, 128, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='C'))
            block_list.append(block(512, 128, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(nn.Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                        padding=(0, 0, 0), bias=False))

        return nn.Sequential(*block_list)


resnet = DMSN(Bottleneck, [3, 4, 6, 3])
# resnet = resnet.to(device=6)
# data = torch.autograd.Variable(
#     torch.rand(8, 3, 16, 112, 112)).to(device=6)  # if modality=='Flow', please change the 2nd dimension 3==>2
# out = resnet(data)
# print(out.size(), out)
# 向网络输入一个1，3，224，224的tensor
x = torch.randn(8, 3, 16, 112, 112)
x = resnet(x)
print(x.shape)