#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Author: xyx
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from audio_model import resnet18_audio
# from visual_model import resnet18
from av_classify2 import AV_Model
from data_loader_dense import *
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#import av_model
from time import strftime, localtime
from backbone.resnet import resnet18
import argparse

def printTime():
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, trainloader, optimizer, epoch, criterion, device, f):
    total = 0
    correct = 0
    print("start training epoch %d"%epoch)
    model.train()
    for batch_idx, data in enumerate(trainloader):
        image_data, label = data
        image_data, label = image_data.type(torch.FloatTensor).to(device), \
                                        label.type(torch.LongTensor).to(device)
        #print('img_size',image_data.size())
        optimizer.zero_grad()
        output = model(image_data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        predicted = output.max(1, keepdim=True)[1]
        total += label.size(0)
        correct += (predicted.view(-1) == label.view(-1)).sum().item()
        if batch_idx % 200 == 0:
            printTime()
            # break
            print("Epoch %d, Batch %5d, loss %.5f" % (epoch, batch_idx, loss.item()))
            f.write("Epoch %d, Batch %5d, loss %.5f\n" % (epoch, batch_idx, loss.item()))
    acc = 100.0 * correct / total
    print('Train Accuracy : %f %% at %d epoch' % (acc, epoch))
    f.write('Train Accuracy : %f %% at %d epoch\n' % (acc, epoch))
    torch.save(model.state_dict(), 'checkpoint/epoch%d.pth'%epoch)

def test(model, test_loader, best_acc, best_epoch, epoch, device):
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            image_data, label = data
            image_data, label = image_data.type(torch.FloatTensor).to(device), \
                                label.type(torch.LongTensor).to(device)
            #print(image_data.size())
            outputs = model(image_data)
            # print(outputs)
            # ???????????????????????????????????????????????????????????????????????????????????????????????????
            predicted = outputs.max(1, keepdim=True)[1]
            total += label.size(0)
            correct += (predicted.view(-1) == label.view(-1)).sum().item()
            # if (batch_idx + 1) % 4 == 0:
            #     printTime()
    acc = 100.0 * correct / total
    if best_acc < acc:
        best_acc = acc
        best_epoch = epoch
    printTime()
    print('Accuracy : %f %% at %d epoch' % (acc, epoch))
    return best_acc, best_epoch


def main():
    parser = argparse.ArgumentParser(description="resLP training")
    parser.add_argument('-m', '--model', type=str, default='r2d_18', help='r2d_18 | attention')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-gd', '--grad_norm', type=float, default=40)
    parser.add_argument('-g', '--gpu_ids', type=str, default='0', help='available gpu ids')
    parser.add_argument('-s', '--seed', type=int, default=999, help='random seed')
    parser.add_argument('-l', '--level', type=str, default='4', help='conv level')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning_rate')
    parser.add_argument('-d', '--dataset', type=str, default='UCF101', help='dataset')
    args = parser.parse_args()

    random_seed = args.seed
    ngpu = 1
    batch_size = args.batch_size
    lr = args.learning_rate
    level = 'level%s'%args.level
    model_name = args.model
    dataset = args.dataset
    gpu_id = args.gpu_ids

    setup_seed(random_seed)
    if model_name == 'r2d_18':
        visual_net = resnet18(pretrained=False, progress=True)
    else:
        print('unknown model')
        visual_net = resnet18(pretrained=False, progress=True)

    av_model = AV_Model(visual_net=visual_net, a_l=level, v_l=level, data=dataset)
    device = torch.device("cuda:%s"%gpu_id if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    model = av_model.to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70, 90], gamma=0.1)

    train_data = Dataset('/home/share/GSAI-M3PL-Lab/Kinetics-Sounds/ks-train-frame-4fps/ks-train-set/', './dataset', 'KS_train.csv')
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, num_workers=16, pin_memory=True)
    test_data = Dataset('/home/share/GSAI-M3PL-Lab/Kinetics-Sounds/ks-test-frame-4fps/test-set/', './dataset',
                                'KS_test.csv')
    test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size, num_workers=16, pin_memory=True)

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(0, 150):
        scheduler.step()
        f = open('log/%s_%s.txt'%(model_name,dataset), 'a', encoding='utf-8')
        train(model, train_loader, optimizer, epoch, criterion, device, f)
        best_acc, best_epoch = test(model, test_loader, best_acc, best_epoch, epoch, device)
        print('Best_Accuracy : %f %% at %d epoch' % (best_acc, best_epoch))
        f.write('Best_Accuracy : %f %% at %d epoch\n' % (best_acc, best_epoch))
        f.close()

main()
