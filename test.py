from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp

import numpy as np
import sys
sys.path.append("..")
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import pdb
import yaml
from easydict import EasyDict as edict

from reid import datasets
from reid.models import model_creator
from reid.metric_learning import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.evaluators_sa import Evaluator_sa
from reid.loss import merge_a_into_b
from reid.datasets.data import transforms as T
from reid.datasets.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint


def check_cfg(args):



    if args.model == 'resnet_mgn' or args.model == 'resnet_mgn_lr' :
        assert args.height == 384
        assert args.width == 128
    else:
        assert args.height == 256
        assert args.width == 128


def get_data(name, split_id, data_dir, height, width, batch_size, 
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_trainval_ids
    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True



    cfg_file = osp.join('.','cfgs', args.model, 'stage'+str(args.stage)+'.yml')
    if os.path.exists(cfg_file):
        print('cfg_file :',cfg_file)
        with open(cfg_file, 'r') as f:
            yaml_cfg = edict(yaml.load(f))
        merge_a_into_b(yaml_cfg, args.__dict__)
    else:
        print('cfg files not exists, using default params')
    check_cfg(args)




    dataset, num_classes, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size,  args.workers) # 32 is batch_size for test

    # Create model
    model = model_creator(args.model, args.depth, pretrained=True, num_features=args.num_features, norm = args.norm,
                          dropout=args.dropout, num_classes=num_classes, add_l3_softmax=False, test_stage=False)#args.features

    #model = nn.DataParallel(model).cuda()
    # Load from checkpoint
    start_epoch = best_map = prior_best_map = 0
    if args.resume_path:
        ckpt_file = args.resume_path
    else:
        logs_base_dir = osp.join('.', 'logs', args.dataset, args.model)
        ckpt_dir = osp.join(logs_base_dir,'stage'+str(args.stage))
        ckpt_file = osp.join(ckpt_dir, 'model_best.pth.tar')

    if not os.path.exists(ckpt_file):
        raise RuntimeError(ckpt_file+'resume model doesnot exist!')
    else:
        print("resume model from", ckpt_file)
        checkpoint = load_checkpoint(ckpt_file)
        model.load_state_dict(checkpoint['state_dict'])
        curr_best_map = checkpoint['best_map']
        print("=> Start epoch {}  best map {:.1%}"
              .format(start_epoch, curr_best_map))

    model = nn.DataParallel(model).cuda()
    model.eval()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # # # Evaluator
    # evaluator = Evaluator(model)
    # # metric.train(model, train_loader)
    # print("Test:")
    # evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)

    evaluator = Evaluator_sa(model)
    # metric.train(model, train_loader)
    print("Test:")
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)

    # evaluator = Evaluator_rerank(model)
    # # metric.train(model, train_loader)
    # print("Test:")
    # evaluator.evaluate_rerank(test_loader, dataset.query, dataset.gallery, metric, if_cmc=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='../dataset')
    parser.add_argument( '--dataset', type=str, default='market1501',
                        choices=['cuhk03', 'market1501', 'dukemtmc','market1501_2'])

    parser.add_argument('--stage', type=int, default= 0)
    parser.add_argument('--model',  type=str, default='resnet_channel3',
                         choices=['resnet_mgn','resnet_mgn_lr','resnet_maxout2','resnet_reid','resnet_channel',
                                  'resnet_channel2','resnet_channel3','resnet_bypath'])
    
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")

    parser.add_argument('--batch_size', type=int, default=160)   # original 72
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    # model
    parser.add_argument('--depth',  type=int, default='50',
                         choices=[50,101,152])
    parser.add_argument('--num_features', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--norm', type=bool, default=True)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    parser.add_argument('--resume_path', type=str, default=None)

    main(parser.parse_args())
