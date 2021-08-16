import argparse
import os
import random
import shutil
from src import utils
import time
from meters import AverageMeter, ProgressMeter, TensorboardMeter
from my_args import get_args
import warnings

from src.prepare_data import prepare_data

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary

# GPU if available (or CPU instead)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO: best accuracy metrics (used to save the best checkpoints)
best_miou = 0.


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # create the experiment dir if it does not exist
    if not os.path.exists(os.path.join('experiments', args.experiments)):
        os.mkdir(os.path.join('experiments', args.experiments))

    # TODO: define model
    model = None

    # define input_size here to have the right summary of your model
    if args.summary:
        summary(model, input_size=(3, 480, 640))
        exit()

    # dataloaders code
    data_loaders = prepare_data(args, ckpt_dir=None)
    train_loader, val_loader = data_loaders

    cameras = train_loader.dataset.cameras
    n_classes_without_void = train_loader.dataset.n_classes_without_void
    if args.class_weighting != 'None':
        class_weighting = train_loader.dataset.compute_class_weights(weight_mode=args.class_weighting, c=args.c_for_logarithmic_weighting)
    else:
        class_weighting = np.ones(n_classes_without_void)

    # loss functions (only loss_function_train is really needed.
    # The other loss functions are just there to compare valid loss to train loss)
    criterion_train = \
        utils.CrossEntropyLoss2d(weight=class_weighting, device=device)

    pixel_sum_valid_data = val_loader.dataset.compute_class_weights(weight_mode='linear')
    pixel_sum_valid_data_weighted = np.sum(pixel_sum_valid_data * class_weighting)
    criterion_val = utils.CrossEntropyLoss2dForValidData(
        weight=class_weighting,
        weighted_pixel_sum=pixel_sum_valid_data_weighted,
        device=device
    )
    criterion_val_unweighted = \
        utils.CrossEntropyLoss2dForValidDataUnweighted(device=device)

    # define optimizer
    optimizer = get_optimizer(args, model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] if args.start_epoch is None else args.start_epoch
            best_miou = checkpoint['best_miou'].to(device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # If only evaluating the model is required
    if args.evaluate:
        _, _, _ = one_epoch(val_loader, model, criterion_val, 0, args, optimizer=None)
        return

    # define tensorboard meter
    tensorboard_meter = TensorboardMeter(f"experiments/{args.experiment}/logs")

    # TRAINING + VALIDATION LOOP
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        miou, loss = one_epoch(train_loader, model, criterion_train,
                               epoch, args, tensorboard_meter, optimizer=optimizer)

        # evaluate on validation set (optimizer is None when validation)
        miou, loss = one_epoch(val_loader, model, criterion_val,
                               epoch, args, tensorboard_meter, optimizer=None)

        # remember best accuracy and save checkpoint
        is_best = miou > best_miou
        best_miou = max(miou, best_miou)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_miou': best_miou,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=f'{args.experiment}/checkpoint_{str(epoch).zfill(5)}.pth.tar')


def one_epoch(dataloader, model, criterion, epoch, args, tensorboard_meter: TensorboardMeter, optimizer=None):
    """One epoch pass. If the optimizer is not None, the function works in training mode. 
    """
    # TODO: define AverageMeters (print some metrics at the end of the epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracies = AverageMeter('Accuracy', ':6.2f')

    is_training = optimizer is not None
    prefix = 'TRAIN' if is_training else 'TEST'

    # TODO: final Progress Meter (add the relevant AverageMeters)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, accuracies],
        prefix=f"{prefix} - Epoch: [{epoch}]")

    # switch to train mode (if training)
    if is_training:
        model.train()
    else:
        model.eval()

    end = time.time()
    for i, (images, target) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # TODO: define accuracy metrics
        accuracy = torch.Tensor(1.1)  # TODO: here
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy[0], images.size(0))

        if is_training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # if debugging, stop after the first batch
        if args.debug:
            break

        # TODO: define AverageMeters used in tensorboard summary
        if is_training:
            tensorboard_meter.update_train([accuracies, losses])
        else:
            tensorboard_meter.update_val([accuracies, losses])

        return accuracies.avg, losses.avg  # TODO


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optimizer(args, model):
    # set different learning rates fo different parts of the model
    # when using default parameters the whole model is trained with the same
    # learning rate
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(
            'Currently only SGD and Adam as optimizers are '
            'supported. Got {}'.format(args.optimizer))

    print('Using {} as optimizer'.format(args.optimizer))
    return optimizer


if __name__ == '__main__':
    main()
