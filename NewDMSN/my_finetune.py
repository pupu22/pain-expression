import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
from torchvision.transforms import transforms
import torch
import argparse
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.optim as optim
import shutil
import numpy as np
from torch.autograd import Variable
import video_transforms
from DMSN_dataset import DMSNDataSet

from my_DMSN import MyDMSNModel, get_optim_policies

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '222.201.134.236'
    os.environ['MASTER_PORT'] = '2088'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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
    DMSNDataSet("./UNBCtrain.txt",
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

def save_temp(state, filename='NewDMSNUnbc.pth.tar'):
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

def reduce_tensor(tensor, world_size):
    # 用于平均所有gpu上的运行结果，比如loss
    # Reduces the tensor data across all machines
    # Example: If we print the tensor, we can get:
    # tensor(334.4330, device='cuda:1') *********************, here is cuda:  cuda:1
    # tensor(359.1895, device='cuda:3') *********************, here is cuda:  cuda:3
    # tensor(263.3543, device='cuda:2') *********************, here is cuda:  cuda:2
    # tensor(340.1970, device='cuda:0') *********************, here is cuda:  cuda:0
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def train(train_loader, net, criterion, optimizer, epoch):
    # net = nn.DataParallel(net, device_ids=[0])

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net = net.train()
    total_loss = 0

    for batch_idx, (i, data) in enumerate(train_loader, 0):
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

        reduced_loss = reduce_tensor(loss.data, world_size)
        total_loss += reduced_loss.item()

        training_loss = (total_loss / (batch_idx + 1))

        torch.cuda.empty_cache()
        # if i % 10 == 0:
        # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        # 'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'training_loss {training_loss:8.5f}\t'
              'lr {lr}'.format(
            epoch, i, len(train_loader), loss=losses, training_loss = training_loss,lr=optimizer.param_groups[0]['lr']))

    print('Finished Training')


def val(val_loader, net, criterion):
    # 指定显卡
    # net = nn.DataParallel(net, device_ids=[0,1])

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
        torch.cuda.empty_cache()
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
    # net = nn.DataParallel(net, device_ids=[0])

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
        torch.cuda.empty_cache()
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


def main(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    torch.manual_seed(18)
    torch.cuda.manual_seed_all(18)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(rank) # 这里设置 device ，后面可以直接使用 data.cuda(),否则需要指定 rank

    model = MyDMSNModel().cuda()
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()

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
    best_MAE = 1000

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
        train_dataset = DMSNDataSet("UNBCtext2.txt",
                        length=16,
                        modality="RGB",
                        image_tmpl="frame{:06d}.jpg",
                        transform=train_transform,
                        data_type='train',
                        index=i)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6,num_works = 4, drop_last = True, sampler=train_sampler)

        test_dataset = DMSNDataSet("UNBCtext2.txt",
                        length=16,
                        modality="RGB",
                        image_tmpl="frame{:06d}.jpg",
                        transform=train_transform,
                        data_type='train',
                        index=i)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6,num_works = 4, drop_last = True, sampler=test_sampler)

        for epoch in range(start_epoch, epochs):
            # adjust_learning_rate(learning_rate, weight_decay, optimizer, epoch)
            print("start")
            train_sampler.set_epoch(epoch)
            train(train_loader, ddp_model, criterion, optimizer, epoch)
            MSE, MAE = test(test_loader, ddp_model, criterion)
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

        is_best = final_MSE.avg < best_MAE
        best_prec1 = min(final_MSE.avg, best_MAE)
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)