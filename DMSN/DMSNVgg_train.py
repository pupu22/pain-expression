import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

import shutil
from Vgg_Dataset import VggDataset
from DMSN import DMSNModel, get_optim_policies

import video_transforms

train_transform = video_transforms.Compose(
    [
        video_transforms.RandomResizedCrop(160),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor(),
        video_transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))]
)
# train_transform = transforms.Compose(
#     [
#         transforms.RandomResizedCrop(160),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(30, resample=False, expand=False, center=None ),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                                    (0.229, 0.224, 0.225))]
# )
val_transform = video_transforms.Compose(
    [
        video_transforms.Resize((160, 160)),
        video_transforms.CenterCrop(160),
        video_transforms.ToTensor(),
        video_transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))]
)

train_loader = torch.utils.data.DataLoader(
    # P3DDataSet("p3dtrain_01.lst",
    VggDataset("VGGFace2train_list.txt",
               length=16,
               modality="RGB",
               image_tmpl="frame{:06d}.jpg",
               transform=train_transform),
    batch_size=10,
    shuffle=True,
    num_workers=16,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    VggDataset("VGGFace2test_list.txt",
               length=16,
               modality="RGB",
               type='val',
               image_tmpl="frame{:06d}.jpg",
               transform=val_transform),
    batch_size=10,
    shuffle=False,
    num_workers=16,
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


def save_temp(state, filename='DMSNTemp.pth.tar'):
    torch.save(state, filename)


def save_checkpoint(state, is_best, filename='DMSNCheckpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'DMSNBest.pth.tar')


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
        # labels = torch.stack(labels, dim=1)
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # inputs,labels=Variable(inputs),Variable(labels)
        # print("epoch：", epoch, "的第", i, "个inputs", inputs.data.size(), "labels", labels.data)


        outputs = net(inputs)
        # print(labels.size())
        # print(outputs.size())
        loss = criterion(outputs, labels)
        # print(loss)

        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 2))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top3.update(prec3[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        # for name, parms in net.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
        #           ' -->grad_value:', parms.grad , 'device:',parms.device)

        if i % 10 == 0:
            # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            # 'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.10f})\t'
                  'lr {lr}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses,
                lr=optimizer.param_groups[0]['lr'],
                top1=top1, top3=top3))

        if i % 1000 == 0:
            save_temp({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            })

        # del inputs, outputs
    torch.cuda.empty_cache()
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
        inputs, labels = Variable(inputs.to(device=6)), Variable(labels.to(device=6))

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 2))
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

        del inputs, outputs
        torch.cuda.empty_cache()

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))

    return top1.avg


def main():
    model = DMSNModel(num_classes = 2)
    # 模型放到哪张显卡上
    model = model.cuda()

    # model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    policies = get_optim_policies(model)
    learning_rate = 0.01
    weight_decay = 0
    optimizer = optim.SGD(policies, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    start_epoch = 0
    epochs = 10

    best_prec1 = 0

    # resume = 'VGGCheckpoint.pth.tar'
    # if os.path.isfile(resume):
    #     checkpoint = torch.load(resume)
    #     start_epoch = checkpoint['epoch']
    #     best_prec1 = checkpoint['best_prec1']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(resume, checkpoint['epoch']))

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(learning_rate, weight_decay, optimizer, epoch)
        print("start")
        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = val(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
    main()
