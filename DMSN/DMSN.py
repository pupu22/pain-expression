import math

import torch
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F


def conv_T(in_planes, out_planes, stride=(1, 1, 1), padding=(0, 0, 1)):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, padding=padding, bias=False)


def conv_S(in_planes, out_planes, stride=(1, 1, 1), padding=(1, 1, 0)):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 1), stride=stride, padding=padding, bias=False)

# def weigth_init(m):
#    if isinstance(m, nn.Conv3d):
#        init.xavier_uniform_(m.weight.data)
#        init.constant_(m.bias.data,0.1)
#    elif isinstance(m, nn.BatchNorm3d):
#        m.weight.data.fill_(1)
#        m.bias.data.zero_()
#    elif isinstance(m, nn.Linear):
#        m.weight.data.normal_(0,0.01)
#        m.bias.data.zero_()

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            init.xavier_uniform(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
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
    def __init__(self, inplane, midplane, StrideS, PaddingS, StrideT, PaddingT, downsample=None, st_struc='A',
                 number=0):
        super(Bottleneck, self).__init__()
        self.st_struc = st_struc
        self.midplane = midplane

        # stage1输入通道数为64，main stage中通道数为他的一半，分支通道数为他的四分之一
        # 这里stage1输入了16，first_plane为32，double_plane为64
        one_plane = midplane * 2

        if self.st_struc == 'A':
            self.conv1 = nn.Conv3d(inplane, inplane, kernel_size=(1, 1, 1),
                                   stride=1, bias=False)
            self.bn1 = nn.BatchNorm3d(inplane)
            self.conv2 = conv_T(inplane, one_plane, stride=StrideT, padding=PaddingT)
            self.bn2 = nn.BatchNorm3d(one_plane)
            self.conv6 = conv_S(one_plane, midplane, stride=StrideS, padding=PaddingS)
            self.bn6 = nn.BatchNorm3d(midplane)

            self.conv3 = conv_T(one_plane, one_plane, stride=(1, 1, 1), padding=(0, 0, 1))
            self.bn3 = nn.BatchNorm3d(one_plane)
            self.conv7 = conv_S(one_plane, midplane, stride=StrideS, padding=PaddingS)
            self.bn7 = nn.BatchNorm3d(midplane)

            self.conv4 = conv_T(one_plane, one_plane, stride=(1, 1, 1), padding=(0, 0, 1))
            self.bn4 = nn.BatchNorm3d(one_plane)
            self.conv8 = conv_S(one_plane, midplane, stride=StrideS, padding=PaddingS)
            self.bn8 = nn.BatchNorm3d(midplane)

            self.conv5 = conv_T(one_plane, one_plane, stride=(1, 1, 1), padding=(0, 0, 1))
            self.bn5 = nn.BatchNorm3d(one_plane)
            self.conv9 = conv_S(one_plane, midplane, stride=StrideS, padding=PaddingS)
            self.bn9 = nn.BatchNorm3d(midplane)

        elif self.st_struc == 'B':
            self.conv1 = nn.Conv3d(inplane, inplane, kernel_size=(1, 1, 1),
                                   stride=1, bias=False)
            self.bn1 = nn.BatchNorm3d(inplane)
            self.conv6 = conv_S(inplane, one_plane, stride=StrideS, padding=PaddingS)
            self.bn6 = nn.BatchNorm3d(one_plane)
            self.conv2 = conv_T(one_plane, midplane, stride=StrideT, padding=PaddingT)
            self.bn2 = nn.BatchNorm3d(midplane)

            self.conv3 = conv_T(one_plane, one_plane, stride=StrideT, padding=PaddingT)
            self.bn3 = nn.BatchNorm3d(one_plane)
            self.conv7 = conv_S(one_plane, midplane, stride=StrideS, padding=PaddingS)
            self.bn7 = nn.BatchNorm3d(midplane)

            self.conv8 = conv_S(one_plane, one_plane, stride=StrideS, padding=PaddingS)
            self.bn8 = nn.BatchNorm3d(one_plane)
            self.conv4 = conv_T(one_plane, midplane, stride=StrideT, padding=PaddingT)
            self.bn4 = nn.BatchNorm3d(midplane)

            self.conv5 = conv_T(one_plane, one_plane, stride=StrideT, padding=PaddingT)
            self.bn5 = nn.BatchNorm3d(one_plane)
            self.conv9 = conv_S(one_plane, midplane, stride=StrideS, padding=PaddingS)
            self.bn9 = nn.BatchNorm3d(midplane)

        elif self.st_struc == 'C':
            self.conv1 = nn.Conv3d(inplane, inplane, kernel_size=(1, 1, 1),
                                   stride=1, bias=False)
            self.bn1 = nn.BatchNorm3d(inplane)
            self.conv6 = conv_S(inplane, one_plane, stride=StrideS, padding=PaddingS)
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

        self.conv10 = nn.Conv3d(midplane * self.extention, midplane * self.extention, kernel_size=1, stride=1)
        self.bn10 = nn.BatchNorm3d(midplane * self.extention)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = 1

    def ST_A(self, T):

        T1 = self.conv2(T)
        T1 = self.bn2(T1)
        T1 = self.relu(T1)
        # print("T1")
        # print(T1.size())
        ST1 = self.conv6(T1)
        ST1 = self.bn6(ST1)
        ST1 = self.relu(ST1)
        # print("ST1")
        # print(ST1.size())
        # ST1 = np.concatenate(())

        T2 = self.conv3(T1)
        T2 = self.bn3(T2)
        T2 = self.relu(T2)
        # print(T2.size())
        ST2 = self.conv7(T2)
        ST2 = self.bn7(ST2)
        ST2 = self.relu(ST2)
        # print(ST2.size())

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
        # print(T4.size())
        ST4 = self.conv9(T4)
        ST4 = self.bn9(ST4)
        ST4 = self.relu(ST4)
        # print(ST4.size())
        # ST4 = np.concatenate(ST3, ST4)
        # ST4 = np.concatenate((ST1.cpu().detach(), ST2.cpu().detach(), ST3.cpu().detach(), ST4.cpu().detach()), axis=1)
        ST4 = torch.cat([ST1, ST2, ST3, ST4], dim=1)
        # ST4 = np.concatenate((ST1, ST2, ST3, ST4), axis=1)
        # ST4 = torch.from_numpy(ST4).to(device=6)

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
        ST4 = torch.cat([ST1, ST2, ST3, ST4], dim=1)
        # ST4 = np.concatenate((ST1.cpu().detach(), ST2.cpu().detach(), ST3.cpu().detach(), ST4.cpu().detach()), axis=1)
        # ST4 = torch.from_numpy(ST4).to(device=6)

        return ST4

    def ST_C(self, S):
        # print("C_input")
        # print(S.size())
        S1 = self.conv6(S)
        S1 = self.bn6(S1)
        S1 = self.relu(S1)
        # print("S1")
        # print(S1.size())
        ST1 = self.conv2(S1)
        ST1 = self.bn2(ST1)
        ST1 = self.relu(ST1)
        # print(ST1.size())

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

        ST4 = torch.cat([ST1, ST2, ST3, ST4], dim=1)
        # ST4 = np.concatenate((ST1.cpu().detach(), ST2.cpu().detach(), ST3.cpu().detach(), ST4.cpu().detach()), axis=1)
        # ST4 = torch.from_numpy(ST4).to(device=6)

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

        if self.downsample is not None:
            residual = self.downsample(xx)

        out = self.conv10(out)
        out = self.bn10(out)
        out = self.relu(out)
        # out = np.concatenate((out.detach(), residual.detach()), axis=1)
        # out = torch.from_numpy(out)
        # if isinstance(out.data, torch.cuda.FloatTensor):
        out = out + residual
        out = self.relu(out)

        return out


class DMSN(nn.Module):

    # 初始化网络结构和参数
    def __init__(self, block, layers, num_classes=6):
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
        self.ConvSize = nn.Conv3d(3, self.inplane, kernel_size=(1, 1, 1), stride=(1, 2, 2), padding=(3, 3, 3),
                                  bias=False)
        self.stage1 = self.make_layer(self.block, 16, self.layers[0], shortcut_type=1, stride=1)
        self.stage2 = self.make_layer(self.block, 32, self.layers[1], shortcut_type=2, stride=2)
        self.stage3 = self.make_layer(self.block, 64, self.layers[2], shortcut_type=3, stride=2)
        self.stage4 = self.make_layer(self.block, 128, self.layers[3], shortcut_type=4, stride=2)

        # 后续的网络
        self.Savgpool = nn.AvgPool3d(kernel_size=(1, 3, 3))
        self.avgpool = nn.AvgPool3d(kernel_size=(5, 1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):

        # stem部分:conv+bn+relu+maxpool
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # block
        # print("--------------------------stage1----------------------")
        out = self.stage1(out)
        # print("--------------------------stage2----------------------")
        out = self.stage2(out)
        # print("--------------------------stage3----------------------")
        out = self.stage3(out)
        # print("--------------------------stage4----------------------")
        out = self.stage4(out)

        # 分类
        out = self.Savgpool(out)
        out = self.avgpool(out)
        out = np.squeeze(out)
        # out = torch.flatten(out, 2)
        out = self.fc(out)

        # print("out:", out, "out.size:", out.size(), "len:", len(out.size()))
        if len(out.size()) == 1:
            out = F.softmax(out, dim=0)
        else:
            out = F.softmax(out, dim=1)

        return out

    def make_layer(self, block, midplane, block_num, shortcut_type, stride=1):
        """
            block:block模块
            midplane：每个模块中间运算的维度，一般等于输出维度/4
            block_num：重复次数
            stride：Conv Block的步长
        """

        block_list = []

        if shortcut_type == 1:
            '''
                -----------------------------stage1-------------------------------------
                A-B-C 
                不修改大小，在C模块增加downsample，即直连部分增加一个卷积，改变通道数
            '''
            downsample = nn.Sequential(
                nn.Conv3d(64, 128, stride=(1, 1, 1), padding=(0, 0, 0), kernel_size=1, bias=False),
                nn.BatchNorm3d(128)
            )
            block_list.append(block(self.inplane, 16, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(block(64, 16, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='B'))
            block_list.append(block(64, 32, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), downsample=downsample, st_struc='C'))

        elif shortcut_type == 2:
            '''
                -----------------------------stage2-------------------------------------
                A-B-C-A
                C块修改图像大小与通道数，并在C模块增加downsample，即直连部分增加一个卷积，改变通道数和大小
                C块的stride与padding修改图像大小，number修改通道数
            '''
            downsample = nn.Sequential(
                nn.Conv3d(128, 256, stride=(2, 2, 2), padding=(4, 0, 0), kernel_size=1, bias=False),
                nn.BatchNorm3d(256)
            )
            block_list.append(block(128, 32, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(block(128, 32, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='B'))
            block_list.append(block(128, 32, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='C'))
            block_list.append(block(128, 64, StrideS=(1, 1, 2), PaddingS=(1, 1, 0), StrideT=(1, 2, 1),
                                    PaddingT=(0, 0, 1), downsample=downsample, st_struc='A', number=1))
            # block_list.append(nn.Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1),
            #                             padding=(0, 0, 0), bias=False))
        elif shortcut_type == 3:
            '''
                -----------------------------stage3-------------------------------------
                A-B-C-A-B-C 
                最后一个C块修改图像大小与通道数，并在C模块增加downsample，即直连部分增加一个卷积，改变通道数和大小
                C块的stride与padding修改图像大小，number修改通道数
            '''
            downsample = nn.Sequential(
                nn.Conv3d(256, 512, stride=(2, 2, 2), padding=(4, 0, 0), kernel_size=1, bias=False),
                nn.BatchNorm3d(512)
            )
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
            block_list.append(block(256, 128, StrideS=(1, 1, 2), PaddingS=(1, 1, 0), StrideT=(1, 2, 1),
                                    PaddingT=(0, 0, 1), downsample=downsample, st_struc='C', number=1))
            # block_list.append(nn.Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1),
            #                             padding=(0, 0, 0), bias=False))
        elif shortcut_type == 4:
            '''
                -----------------------------stage4-------------------------------------
                A-B-C-A
                最后一个A块修改图像大小与通道数，并增加downsample，即直连部分增加一个卷积，改变通道数和大小
                C块的stride与padding修改图像大小，number修改通道数
            '''
            downsample = nn.Sequential(
                nn.Conv3d(512, 1024, stride=(2, 2, 2), padding=(4, 0, 0), kernel_size=1, bias=False),
                nn.BatchNorm3d(1024)
            )
            block_list.append(block(512, 128, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='A'))
            block_list.append(block(512, 128, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='B'))
            block_list.append(block(512, 128, StrideS=(1, 1, 1), PaddingS=(1, 1, 0), StrideT=(1, 1, 1),
                                    PaddingT=(0, 0, 1), st_struc='C'))
            block_list.append(block(512, 256, StrideS=(1, 1, 2), PaddingS=(1, 1, 0), StrideT=(1, 2, 1),
                                    PaddingT=(0, 0, 1), downsample=downsample, st_struc='A', number=1))
            # block_list.append(nn.Conv3d(512, 1024, kernel_size=(1, 1, 1), stride=(1, 1, 1),
            #                             padding=(0, 0, 0), bias=False))

        return nn.Sequential(*block_list)


def DMSNModel( pretrained = False, **kwargs):
    model = DMSN(Bottleneck, [3, 4, 6, 3], num_classes=6)
    initNetParams(model)
    if pretrained == True:
        pretrained_file = '/home/cike/pythonGC/DMSNBest.pth.tar'
        pretrained_dict = torch.load(pretrained_file)
        weights = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in weights and 'fc' not in k)}
        weights.update(pretrained_dict)
        model.load_state_dict(weights)
    return model


def get_optim_policies(model=None, modality='RGB', enable_pbn=True):
    '''
    first conv:         weight --> conv weight
                        bias   --> conv bias
    normal action:      weight --> non-first conv + fc weight
                        bias   --> non-first conv + fc bias
    bn:                 the first bn3, and many all bn2.

    '''
    first_conv_weight = []
    first_conv_bias = []
    normal_weight = []
    normal_bias = []
    bn = []

    if model == None:
        log.l.info('no model!')
        exit()

    conv_cnt = 0
    bn_cnt = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_weight.append(ps[0])
            if len(ps) == 2:
                normal_bias.append(ps[1])

        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_cnt += 1
            # later BN's are frozen
            if not enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
        elif isinstance(m, torch.nn.BatchNorm2d):
            bn.extend(list(m.parameters()))
        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    slow_rate = 0.7
    n_fore = int(len(normal_weight) * slow_rate)
    slow_feat = normal_weight[:n_fore]  # finetune slowly.
    slow_bias = normal_bias[:n_fore]
    normal_feat = normal_weight[n_fore:]
    normal_bias = normal_bias[n_fore:]

    return [
        {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1, 'decay_mult': 1,
         'name': "first_conv_weight"},
        {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2, 'decay_mult': 0,
         'name': "first_conv_bias"},
        {'params': slow_feat, 'lr_mult': 1, 'decay_mult': 1,
         'name': "slow_feat"},
        {'params': slow_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "slow_bias"},
        {'params': normal_feat, 'lr_mult': 1, 'decay_mult': 1,
         'name': "normal_feat"},
        {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
         'name': "normal_bias"},
        {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
         'name': "BN scale/shift"},
    ]


# resnet = DMSN(Bottleneck, [3, 4, 6, 3])
# resnet = resnet.to(device = 6)
# data = torch.autograd.Variable(
#     torch.rand(8, 3, 16, 112, 112)).to(device = 6) # if modality=='Flow', please change the 2nd dimension 3==>2
# out = resnet(data)
# print(out.size(), out)

# 向网络输入一个1，3，224，224的tensor
# x = torch.randn(8, 3, 16, 112, 112)
# x = resnet(x)
# print(x.shape)

# loss_func = nn.CrossEntropyLoss()
# x = torch.rand((1, 4))
# y = torch.tensor([0])
# z = loss_func(x, y)
# print(z)
