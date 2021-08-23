#!/usr/bin/env python

from collections import OrderedDict
import argparse
import importlib
import json
import logging
import pathlib
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torchvision
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import *
from transformations.image.mixtures.utils import *

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')

def arg_pairs(inputs):
    try:
        x, y = map(int, inputs.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Pairs must be formated as: a,b x,y ...")

class CEwST:
    def __init__(self,  reduction='mean'):
        self.reduction = reduction

    def __call__(self, logits, target):
        """
        Cross Entropy with Soft Target (CEwST) Loss
        :param logits: (batch, *)
        :param target: (batch, *) same shape as logits, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        logprobs = torch.nn.functional.log_softmax(logits.view(logits.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        if self.reduction == 'none':
            return batchloss
        elif self.reduction == 'mean':
            return torch.mean(batchloss)
        elif self.reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset config
    parser.add_argument('--dataset',type=str,default='CIFAR10',choices=['CIFAR10', 'MNIST'])
    parser.add_argument('--use_basicaug', action='store_true')
    parser.add_argument('--use_randaug', action='store_true')
    parser.add_argument('--use_mixup2', action='store_true')
    parser.add_argument('--use_cutmix2', action='store_true')
    parser.add_argument('--use_tile', action='store_true')
    parser.add_argument('--use_reducemix', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--target_pairs', type=arg_pairs, nargs='*',
                        help='Enter: 0,1 2,8 s,t \
                        to cut class 1 into class 0  and class 8 into class 2 \
                        and so on for t into s', default=[])
    parser.add_argument('--target_prob', type=float, default=1.0)
    parser.add_argument('--resize_prob', type=float, default=0.0)
    parser.add_argument('--num_tiles', type=int, default=4)
    parser.add_argument('--randaug_n', type=int, default=2)
    parser.add_argument('--randaug_m', type=int, default=3)
    parser.add_argument('--update_targets', action='store_true')

    # model config
    parser.add_argument('--block_type',type=str,default='basic',choices=['basic', 'bottleneck'])
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--input_shape', default=[1,3,32,32], nargs=4, 
                        metavar=('batch_size', 'color_channels', 'height', 'width'),
                        type=int, help='specify the shape of the input')
    parser.add_argument('--n_classes', type=int, default=10)

    # run config
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--continue_train', action='store_true')

    # optim config
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--scheduler',type=str,default='cosine',choices=['multistep', 'cosine'])
    parser.add_argument('--milestones', type=str, default='[150, 225]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    # TensorBoard
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false')

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    if args.dataset == 'MNIST':
        args.input_shape = [1,1,28,28]

    if args.use_reducemix:
        args.num_classes += 1

    data_config = OrderedDict([
        ('dataset', args.dataset),
        ('batch_size', args.batch_size),
        ('num_workers', args.num_workers),
        ('n_classes', args.n_classes),
        ('use_basicaug', args.use_basicaug),
        ('use_randaug', args.use_randaug),
        ('use_mixup2', args.use_mixup2),
        ('use_cutmix2', args.use_cutmix2),
        ('use_tile', args.use_tile),
        ('use_reducemix', args.use_reducemix),
        ('alpha', args.alpha),
        ('target_pairs', args.target_pairs),
        ('target_prob', args.target_prob),
        ('resize_prob', args.resize_prob),
        ('num_tiles', args.num_tiles),
        ('randaug_n', args.randaug_n),
        ('randaug_m', args.randaug_m),
    ])

    model_config = OrderedDict([
        ('arch', 'resnet_preact'),
        ('block_type', args.block_type),
        ('depth', args.depth),
        ('base_channels', args.base_channels),
        ('input_shape', tuple(args.input_shape)),
        ('use_reducemix', args.use_reducemix),
        ('n_classes', args.n_classes),
    ])

    run_config = OrderedDict([
        ('seed', args.seed),
        ('outdir', args.outdir),
        ('num_workers', args.num_workers),
        ('device', args.device),
        ('tensorboard', args.tensorboard),
        ('use_tile', args.use_tile),
        ('num_tiles', args.num_tiles),
        ('update_targets', args.update_targets),
        ('continue_train', args.continue_train),
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('nesterov', args.nesterov),
        ('scheduler', args.scheduler),
        ('milestones', json.loads(args.milestones)),
        ('lr_decay', args.lr_decay),
    ])

    config = OrderedDict([
        ('data_config', data_config),
        ('model_config', model_config),
        ('run_config', run_config),
        ('optim_config', optim_config),
    ])

    return config


def load_model(config):
    module = importlib.import_module(config['arch'])
    Network = getattr(module, 'Network')
    return Network(config)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def train(epoch, model, optimizer, criterion, train_loader, run_config, writer):
    
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        if run_config['tensorboard'] and step == 0:
            image = torchvision.utils.make_grid(
                data, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        num = data.size(0)

        loss_ = loss.item()
        loss_meter.update(loss_, num)

        if len(targets.shape) > 1:
            k=2
            if run_config['use_tile']:
                k = run_config['num_tiles']
            y_weights, y_idx = torch.topk(targets, k=k, dim=1)
            out_weights, out_idx = torch.topk(outputs, k=k, dim=1)
            correct_ = torch.sum(torch.eq(y_idx, out_idx) * y_weights)
            accuracy = correct_ / num
        else:
            _, preds = torch.max(outputs, dim=1)
            correct_ = preds.eq(targets).sum().item()
            accuracy = correct_ / num

        accuracy_meter.update(accuracy, num)

        if run_config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            accuracy_meter.val,
                            accuracy_meter.avg,
                        ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)

    accuracy = accuracy_meter.avg
    if type(accuracy) != float:
        accuracy = accuracy.item()
    train_log = OrderedDict({
        'epoch': epoch,
        'train': OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': accuracy,
            'time': elapsed,
        }),  
    })
    return train_log


def test(epoch, model, criterion, test_loader, run_config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            if run_config['tensorboard'] and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image('Test/Image', image, epoch)

            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            correct_ = preds.eq(targets).sum().item()
            correct_meter.update(correct_, 1)

            num = data.size(0)

            loss_ = loss.item()
            loss_meter.update(loss_, num)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    test_log = OrderedDict({
        'epoch': epoch,
        'test': OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': accuracy,
            'time': elapsed,
        }),
    })
    return test_log


def main():
    # parse command line arguments
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    device = torch.device(run_config['device'])

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = pathlib.Path(run_config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    # TensorBoard SummaryWriter
    writer = SummaryWriter(
        outdir.as_posix()) if run_config['tensorboard'] else None

    # save config as json file in output directory
    outpath = outdir / 'config.json'
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    # data loaders
    train_loader, test_loader = get_loader(data_config)

    # model
    model = load_model(config['model_config'])
    model.to(torch.device(run_config['device']))
    if run_config['continue_train']:
        model_state = torch.load(run_config['outdir'] + '/model_state.pth')
        model.load_state_dict(model_state['state_dict'])
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))

    train_criterion = nn.CrossEntropyLoss(reduction='mean')
    if data_config['use_mixup2'] or data_config['use_cutmix2'] or data_config['use_tile']:
        train_criterion = CEwST(reduction='mean')
    test_criterion  = nn.CrossEntropyLoss(reduction='mean')

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optim_config['base_lr'],
        momentum=optim_config['momentum'],
        weight_decay=optim_config['weight_decay'],
        nesterov=optim_config['nesterov'])
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, optim_config['epochs'], 0)

    # run test before start training
    test(0, model, test_criterion, test_loader, run_config, writer)

    epoch_logs = []
    best_acc = 0
    for epoch in range(1, optim_config['epochs'] + 1):        

        train_log = train(epoch, model, optimizer, train_criterion,
                          train_loader, run_config, writer)
        test_log = test(epoch, model, test_criterion, test_loader, run_config,
                        writer)

        if run_config['update_targets']:
            print('updating targets...')
            cnf_mat = get_confusion_matrix(
                model, 
                test_loader, 
                device, 
                normalize=True
            )
            if not data_config['use_tile']:
                new_targets = get_most_confused_per_class(cnf_mat)
            else:
                new_targets = get_k_most_confused_per_class(cnf_mat, data_config['num_tiles'] - 1)
            print('old targets:', data_config['target_pairs'])
            print('new targets:', new_targets)
            data_config['target_pairs'] = new_targets
            train_loader.collate_fn.target_pairs = new_targets

        scheduler.step()

        epoch_log = train_log.copy()
        epoch_log.update(test_log)
        epoch_logs.append(epoch_log)

        with open(outdir / 'log.json', 'w') as fout:
            json.dump(epoch_logs, fout, indent=2)

        state = OrderedDict([
            ('config', config),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('accuracy', test_log['test']['accuracy']),
        ])

        if test_log['test']['accuracy'] > best_acc:
            best_acc = test_log['test']['accuracy']
            model_path = outdir / 'model_state.pth'
            torch.save(state, model_path)
            
    print('best_acc', best_acc)

if __name__ == '__main__':
    main()
