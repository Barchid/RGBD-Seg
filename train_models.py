from models.stdcseg import BiSeNet
from metrics import M_IOU, ConfusionMatrix
import os
import random
import shutil
import time
from meters import AverageMeter, ProgressMeter, TensorboardMeter
from my_args import get_args
import warnings

from prepare_data import prepare_data

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
from torchvision.models import segmentation

# GPU if available (or CPU instead)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO: best accuracy metrics (used to save the best checkpoints)
best_miou = 0.


def main():
    args = get_args()
    args.valid_full_res = False
    args.batch_size_valid = args.batch_size
    args.c_for_logarithmic_weighting = 1.02

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
    if not os.path.exists(os.path.join('experiments', args.experiment)):
        os.mkdir(os.path.join('experiments', args.experiment))

    # dataloaders code
    data_loaders = prepare_data(args, ckpt_dir=None)
    train_loader, val_loader = data_loaders

    # if in debugging mode, the model is trained on the first batch of the validation set (because there is no shuffle)
    train_loader = val_loader if args.debug else train_loader

    cameras = train_loader.dataset.cameras
    n_classes_without_void = train_loader.dataset.n_classes_without_void
    if args.class_weighting != 'None':
        class_weighting = train_loader.dataset.compute_class_weights(weight_mode=args.class_weighting, c=args.c_for_logarithmic_weighting)
    else:
        class_weighting = np.ones(train_loader.dataset.n_classes_without_void)

    # TODO: define model
    model = BiSeNet(backbone='STDCNet1446', n_classes=train_loader.dataset.n_classes)
    model.to(device)

    # define input_size here to have the right summary of your model
    if args.summary:
        summary(model, input_size=(3, 480, 640))
        exit()

    # loss functions (only loss_function_train is really needed.
    # The other loss functions are just there to compare valid loss to train loss)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index).to(device)
    criterion_f16 = nn.CrossEntropyLoss(ignore_index=args.ignore_index).to(device)
    criterion_f32 = nn.CrossEntropyLoss(ignore_index=args.ignore_index).to(device)

    # define optimizer
    optimizer = get_optimizer(args, model)

    # instantiate the confusion matrix used to compute the accuracy and mIoU metrics
    confmat = ConfusionMatrix(num_classes=train_loader.dataset.n_classes, average=None)

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
        with torch.no_grad():
            # criterion_val.reset_loss()
            _, _, _ = one_epoch(val_loader, model, criterion, 0, confmat, args, optimizer=None, criterion_f16=criterion_f16, criterion_f32=criterion_f32)
        return

    # define tensorboard meter
    tensorboard_meter = TensorboardMeter(f"experiments/{args.experiment}/logs")

    # TRAINING + VALIDATION LOOP
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        miou, loss = one_epoch(train_loader, model, criterion, epoch, confmat, args, tensorboard_meter, optimizer=optimizer, criterion_f16=criterion_f16, criterion_f32=criterion_f32)

        # jump to next epoch if debugging mode
        if args.debug:
            continue

        # evaluate on validation set (optimizer is None when validation)
        with torch.no_grad():
            miou, loss = one_epoch(val_loader, model, criterion, epoch, confmat, args, tensorboard_meter, optimizer=None, criterion_f16=criterion_f16, criterion_f32=criterion_f32)

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


def one_epoch(dataloader, model, criterion, epoch, confmat: ConfusionMatrix, args, tensorboard_meter: TensorboardMeter = None, optimizer=None, criterion_f16=None, criterion_f32=None):
    """One epoch pass. If the optimizer is not None, the function works in training mode. 
    """
    # define AverageMeters (print some metrics at the end of the epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.4f')
    accuracies = AverageMeter('Accuracies', ':6.2f', avg_as_val=True)
    mious = AverageMeter('mIoU', ':6.2f', avg_as_val=True)

    is_training = optimizer is not None
    prefix = 'TRAIN' if is_training else 'TEST'

    # final Progress Meter (add the relevant AverageMeters)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, accuracies, mious],
        prefix=f"{prefix} - Epoch: [{epoch}]")

    # switch to train mode (if training)
    if is_training:
        model.train()
    else:
        model.eval()

    # create confusion matrix to compute the miou and accuracy
    confmat.reset()  # reset the confusion matrix before using it

    end = time.time()
    for i, sample in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # load the data and send them to gpu
        images = sample['image'].to(device)
        depths = sample['depth'].to(device)
        targets = sample['label'].to(device)
        targets = targets.long()

        # compute output
        if args.modality == 'rgb':
            output, feat_16, feat_32 = model(images)
        else:
            output = model(images, depths)

        # compute gradient and do optimization step
        loss_out = criterion(output, targets)
        loss_f16 = criterion_f16(feat_16, targets)
        loss_f32 = criterion_f32(feat_32, targets)

        loss = loss_out + loss_f16 + loss_f32

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        confmat.update((output, targets))

        losses.update(loss.item(), images.size(0))
        mious.update(confmat.miou(ignore_index=args.ignore_index))
        accuracies.update(confmat.accuracy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # if debugging, stop after the first batch
        if args.debug:
            break

        # define AverageMeters used in tensorboard summary
        if is_training:
            tensorboard_meter.update_train([mious, accuracies, losses])
        else:
            tensorboard_meter.update_val([mious, accuracies, losses])

    return mious.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
