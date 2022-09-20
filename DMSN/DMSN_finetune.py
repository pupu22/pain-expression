import os

# # 只用3号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import shutil
import numpy as np
from torch.optim import lr_scheduler
from torch.autograd import Variable
import video_transforms

from DMSN_dataset import DMSNDataSet
from DMSN import DMSNModel, get_optim_policies

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

train_transform = video_transforms.Compose(
    [
        video_transforms.RandomResizedCrop(160),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor(),
        video_transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))]
)
# video_transforms.RandomResizedCrop(160),
# video_transforms.RandomHorizontalFlip(),
# video_transforms.ToTensor(),
# video_transforms.Normalize((0.485, 0.456, 0.406),
#                            (0.229, 0.224, 0.225))]

val_transform = video_transforms.Compose(
    [
        video_transforms.ToTensor(),
        video_transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))]
)

val_loader = torch.utils.data.DataLoader(
    DMSNDataSet("UNBCtrain.txt",
                length=16,
                modality="RGB",
                image_tmpl="frame{:06d}.jpg",
                transform=val_transform),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_temp(state, filename='DMSNUnbc.pth.tar'):
    torch.save(state, filename)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best.pth.tar')


def adjust_learning_rate(learning_rate, weight_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']


def train(train_loader, net, criterion, optimizer, epoch):
    net = nn.DataParallel(net, device_ids=[0])

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net = net.train()

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # 数据放到哪张显卡上

        # inputs, labels = Variable(inputs), Variable(labels)

        inputs = inputs.cuda()
        labels = labels.cuda()
        # inputs,labels=Variable(inputs),Variable(labels)

        outputs = net(inputs)
        # outputs = torch.unsqueeze(outputs, 0)

        loss = criterion(outputs, labels)

        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 3))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        if i % 10 == 0:
            # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            # 'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lr {lr}'.format(
                epoch, i, len(train_loader), loss=losses,
                lr=optimizer.param_groups[0]['lr']))

    print('Finished Training')


def val(val_loader, net, criterion):
    # 指定显卡
    net = nn.DataParallel(net, device_ids=[0])

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net = net.eval()

    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        # 数据放到哪张显卡上

        inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        # outputs = torch.unsqueeze(outputs, 0)
        loss = criterion(outputs, labels)

        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 3))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        if i % 10 == 0:
            print('Val: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                i, len(val_loader), loss=losses,
                top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))

    return top1.avg


def test(test_loader, net, criterion):
    # 指定显卡
    net = nn.DataParallel(net, device_ids=[0])

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    MAE = AverageMeter()
    MSE = AverageMeter()

    net = net.eval()

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        # 数据放到哪张显卡上
        inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)

        # outputs = torch.unsqueeze(outputs, 0)
        print("outsize",outputs.size())
        loss = criterion(outputs, labels)

        mse_item = np.sum((labels.data.cpu().numpy() - outputs.data.argmax().cpu().numpy()) ** 2) / len(outputs.data)
        mae_item = np.sum(np.absolute(labels.data.cpu().numpy() - outputs.data.argmax().cpu().numpy())) / len(
            outputs.data)
        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 3))
        losses.update(loss.item(), inputs.size(0))
        MSE.update(mse_item)
        MAE.update(mae_item)

        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        if i % 10 == 0:
            # print('Test: [{0}/{1}]\t'
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                  'MSE {MSE.val:.3f} ({MSE.avg:.3f})\t'
                  'MAE {MAE.val:.3f} ({MAE.avg:.3f})'.format(
                i, len(test_loader), loss=losses, top1=top1,
                top3=top3, MSE=MSE, MAE=MAE))

    # print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))
    print(' * MSE {MSE.avg:.3f} MAE {MAE.avg:.3f}'.format(MSE=MSE, MAE=MAE))

    return MSE.avg, MAE.avg


def main():
    model = DMSNModel().cuda()
    # 模型放到哪张显卡上
    # model = nn.DataParallel(model, device_ids=[0])
    # model = model.to(device=7)
    criterion = nn.CrossEntropyLoss().cuda()

    # criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    policies = get_optim_policies(model)
    learning_rate = 0.001
    optimizer = optim.SGD(policies, lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    start_epoch = 0
    epochs = 50
    total_MSE = AverageMeter()
    total_MAE = AverageMeter()
    final_MSE = AverageMeter()
    final_MAE = AverageMeter()
    best_prec1 = 1000

    # resume = 'checkpoint.pth.tar'
    # if os.path.isfile(resume):
    #     checkpoint = torch.load(resume)
    #     start_epoch = checkpoint['epoch']
    #     best_prec1 = checkpoint['best_prec1']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(resume, checkpoint['epoch']))
    for i in range(6):
        print(i)
        train_loader = torch.utils.data.DataLoader(
            # P3DDataSet("p3dtrain_01.lst",
            DMSNDataSet("UNBCtext.txt",
                        length=16,
                        modality="RGB",
                        image_tmpl="frame{:06d}.jpg",
                        transform=train_transform,
                        data_type='train',
                        index=i),
            batch_size=6,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            DMSNDataSet("UNBCtext.txt",
                        length=16,
                        modality="RGB",
                        image_tmpl="frame{:06d}.jpg",
                        transform=train_transform,
                        data_type='test',
                        index=i),
            batch_size=6,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )
        for epoch in range(start_epoch, epochs):
            # adjust_learning_rate(learning_rate, weight_decay, optimizer, epoch)
            print("start")
            train(train_loader, model, criterion, optimizer, epoch)
            MSE, MAE = test(test_loader, model, criterion)
            total_MSE.update(MSE)
            total_MAE.update(MAE)
            torch.cuda.empty_cache()
            print('total_MSE {total_MSE.val:.3f}({total_MSE.avg:.3f})\t'
                  'total_MAE {total_MAE.val:.3f}({total_MAE.avg:.3f})'.format(
                total_MSE=total_MSE, total_MAE=total_MAE))
            save_temp({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })


        # 留一主题交叉验证的每一次的MSE和MAE(val)，以及他们的平均值(avg)
        final_MSE.update(total_MSE.avg)
        final_MAE.update(total_MAE.avg)
        print('final_MSE {final_MSE.val:.3f}({final_MSE.avg:.3f})\t'
              'final_MAE {final_MAE.val:.3f}({final_MAE.avg:.3f})'.format(
            final_MSE=final_MSE, final_MAE=final_MAE))

        is_best = final_MSE.avg < best_prec1
        best_prec1 = min(final_MSE.avg, best_prec1)
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
