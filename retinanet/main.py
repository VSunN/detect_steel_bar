import time
import os
import copy
import argparse
import pdb
import collections
import sys
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
from tensorboardX import SummaryWriter

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):

    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default="csv",help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', default = "./data/train_only.csv",help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default = "./data/classes.csv",help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', default = "./data/train_only.csv",help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--output', help='output dir', type=str, default='./output/000')
    parser.add_argument('--iou_th', help='iou threshold to qualify as detected', type=int, default=0.7)
    parser.add_argument('--conf_th', help='object confidence threshold', type=int, default=0.5)
    parser.add_argument('--nms_th', help='iou threshold for non-maximum suppression', type=int, default=0.4)
    parser.add_argument('--max_detections', help='max detection boxes', type=int, default=300)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')


        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(),pretrained=True, nms_th=parser.nms_th)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True,mode="max")
    #scheduler = optim.lr_scheduler.StepLR(optimizer,8)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    # save output
    output_dir = parser.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file = open(os.path.join(output_dir,'log.txt'),"w")
    loss_dir = os.path.join(output_dir, 'loss')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    writer = SummaryWriter(loss_dir)
    checkpoint_dir = os.path.join(output_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print('Num training images: {}'.format(len(dataset_train)))
    best_map = 0
    print("Training models...")
    for epoch_num in range(parser.epochs):

        #scheduler.step(epoch_num)	
        retinanet.train()
        retinanet.module.freeze_bn()
        
        epoch_loss = []
        
        for iter_num, data in enumerate(dataloader_train):
            try:
                #print(csv_eval.evaluate(dataset_val[:20], retinanet)[0])
                #print(type(csv_eval.evaluate(dataset_val, retinanet)))
                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss
                
                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                if iter_num % 10 == 0:
                    niter = epoch_num * len(dataloader_train) + iter_num
                    writer.add_scalar('Train/epoch_loss', float(loss), niter)
                    writer.add_scalar('Train/running_loss', np.mean(loss_hist), niter)
                    writer.add_scalar('Train/classification_loss', float(classification_loss), niter)
                    writer.add_scalar('Train/regression_loss', float(regression_loss), niter)
                if iter_num % 50 == 0:
                    print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                    log_file.write('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} \n'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet, iou_threshold=parser.iou_th,
                                    score_threshold=parser.conf_th, max_detections=parser.max_detections)
        
        try:
            is_best_map = mAP[0][0] > best_map
            best_map = max(mAP[0][0],best_map)
        except:
            pass
        if is_best_map:
            print("Get better map: ",best_map)
        
            torch.save(retinanet.module, os.path.join(checkpoint_dir, '{}_scale15_{}.pt').format(epoch_num,best_map))
            shutil.copyfile(os.path.join(checkpoint_dir, '{}_scale15_{}.pt').format(epoch_num,best_map),
                            os.path.join(checkpoint_dir, 'best_model.pt'))
        else:
            print("Current map: ",best_map)
        scheduler.step(best_map)
    retinanet.eval()

    torch.save(retinanet, os.path.join(output_dir, 'model_final.pt'))

if __name__ == '__main__':
 main()
