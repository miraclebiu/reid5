from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import pprint

import numpy as np
import sys
sys.path.append("..")
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from  torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, LambdaLR

import pdb
import yaml
from bisect import bisect_right
from easydict import EasyDict as edict

from reid import datasets
from reid.models import model_creator
from reid.metric_learning import DistanceMetric
from reid.loss import loss_creator, merge_a_into_b, loss_config
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.datasets import get_data
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def check_cfg(args):
    if args.trip or args.pnsm_weight or args.quad:
        assert args.sample_instance == True
        assert args.batch_size % args.num_instances == 0, \
            'num_instances should divide batch_size'

    if args.trip :
        assert args.pnsm_weight == False
        assert args.quad == False
        assert args.rank_loss == False
    elif args.pnsm_weight:
        assert args.quad == False
        assert args.rank_loss == False
    elif args.quad:
        assert args.rank_loss == False
    elif not args.rank_loss:
        assert args.xen == True

    if args.model == 'resnet_mgn' or args.model == 'resnet_mgn_lr' or args.model == 'resnet_bypath' or args.model =='resnet_channel2'or args.model =='resnet_channel3':
        assert args.loss_type == 'multi'
    else:
        assert args.loss_type == 'single'

    if args.model == 'resnet_mgn' or args.model == 'resnet_mgn_lr' :
        assert args.height == 384
        assert args.width == 128
        assert args.sac == False
    else:
        assert args.height == 256
        assert args.width == 128


    if args.resume:
        assert args.frozen_sublayer == False

    # if args.frozen_sublayer:
    #     assert args.stage == 0
    #     assert args.xen == True

    # if args.stage == 0:
    #     assert args.frozen_sublayer==True
    # else:
    #     assert args.frozen_sublayer == False



def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    #### load configures from yml file to args  then merge config from args to loss_config
    cfg_file = osp.join('.','cfgs', args.model, 'stage'+str(args.stage)+'.yml')
    if os.path.exists(cfg_file):
        print('cfg_file :',cfg_file)
        with open(cfg_file, 'r') as f:
            yaml_cfg = edict(yaml.load(f))
        merge_a_into_b(yaml_cfg, args.__dict__)
    else:
        print('cfg files not exists, using default params')
    check_cfg(args)

    #### create logs dir and save log.txt
    logs_base_dir = osp.join('.','logs',args.dataset,args.model)
    logs_dir = osp.join(logs_base_dir,'stage'+str(args.stage))
    print("log_dir : ",logs_dir)

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(logs_dir, 'log.txt'))
        print('log start!')
    print("Current experiments parameters")
    pprint.pprint(args)

    

    ##### Create data loaders
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval, args.sample_instance)

    ##### Create model
    model = model_creator(args.model, args.depth, pretrained=True, num_features=args.num_features, norm = args.norm,
                          dropout=args.dropout, num_classes=num_classes, add_l3_softmax=False,test_stage=False)#args.features
    #### resume models
    start_epoch = curr_best_map = 0
    if args.resume:
        if args.resume_path:
            ckpt_file = args.resume_path
        else:
            ckpt_dir = osp.join(logs_base_dir,'stage'+str(args.stage-1))
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
        args.frozen_sublayer = False

    model = nn.DataParallel(model).cuda()

    ##### Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    ##### Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        metric.train(model, train_loader)
        print("Before Test the model:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)

    merge_a_into_b(args,loss_config)


    print("Current loss config")
    print(loss_config)

    print("\n manual check configs to continue experiments !!!\n")
    pdb.set_trace()


    ##### create loss
    criterion = loss_creator(loss_config).cuda()

    # frozen_layerName = ['conv1', 'bn1', 'relu','maxpool', 'layer1', 'layer2',
    #                     'layer3','layer3_0','layer3_1','layer3_2',
    #                     'layer4','layer4_0','layer4_1','layer4_2']
    # frozen_layerName = ['conv1', 'bn1', 'relu','maxpool', 'layer1', 'layer2',
    #                     'layer3','layer3_0','layer3_1','layer3_2',]
    frozen_layerName = ['conv1', 'bn1', 'relu','maxpool', 'layer1', 'layer2',]
    ##### Optimizer
    if args.frozen_sublayer:
        frozen_Source = None
        if hasattr(model.module,'base'):
            frozen_Source = 'model.module.base.'
        elif hasattr(model.module,frozen_layerName[0]):
            frozen_Source = 'model.module.'
        else:
            raise RuntimeError('Not freeze layers but frozen_sublayer is True!')

        base_params_set = set()
        for subLayer in frozen_layerName:
            if hasattr( eval(frozen_Source[:-1]),subLayer):
                print('frozen layer: ', subLayer )
                single_module_param = eval(frozen_Source + subLayer + '.parameters()')
                # base_params.append(single_module_param)
                single_module_param_set = set(map(id, single_module_param))
                base_params_set = base_params_set | single_module_param_set
            else:
                print("current model doesn't have ",subLayer)

        new_params = [p for p in model.parameters() if
                      id(p) not in base_params_set]

        base_params = [p for p in model.parameters() if
                       id(p) in base_params_set]
        param_groups = [
            {'params': base_params, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}
        ]
    else:
        param_groups = model.parameters()

    if args.optimizer =="sgd":
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    else :
        optimizer = torch.optim.Adam(param_groups, lr=args.lr ,
                                 weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion)
    step_epoch = 80
    # Schedule learning rate

    if args.optimizer == 'sgd':
        step_epoch = 80
        sceduler = StepLR(optimizer, step_size=step_epoch, gamma = 0.1)

    else :
        #### the original lr is multiply 10 to start and this is done in creating the optimizer
        step_epoch = 90
        lambda1 = lambda epoch: 0.1 * 0.1 ** ((epoch - step_epoch) / float(step_epoch))
        sceduler =  LambdaLR(optimizer, lr_lambda=lambda1)

    def save_model_when_running(curr_best_map):
        metric.train(model,train_loader)
        top_map = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        is_best = top_map > curr_best_map
        curr_best_map = max(top_map, curr_best_map)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_map': top_map,
        }, is_best, fpath=osp.join(logs_dir, 'checkpoint.pth.tar'))
        return curr_best_map



    evaluate_steps =[i for i in range(1,step_epoch*2) if i%10==0] + [i for i in range(step_epoch*2,args.epochs+1,3)]
        
    # Start training
    for epoch in range(start_epoch, args.epochs):
        sceduler.step()
        print('Current  epoch lr:',optimizer.param_groups[0].get('lr'))
        trainer.train(epoch, train_loader, optimizer)
        if epoch in evaluate_steps:
            curr_best_map = save_model_when_running(curr_best_map)


    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reid experiments")
    ################ decide which yml to load or use default params ###########################
    parser.add_argument('--stage', type=int, default= 1)
    parser.add_argument('--model',  type=str, default='resnet_channel3',
                         choices=['resnet_mgn','resnet_mgn_lr','resnet_maxout2','resnet_reid',
                                  'resnet_channel','resnet_channel2','resnet_channel3','resnet_bypath'])

    ################ default parameters ################################################################

    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='../dataset')
    parser.add_argument( '--dataset', type=str, default='market1501',
                        choices=['cuhk03', 'market1501', 'dukemtmc','market1501_2'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--sample_instance', type=bool, default=False,
                        help ="define using softmax sample or perons*imgs/per_person" )
    parser.add_argument('--combine_trainval', type=bool,default=True,
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num_instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 8")
    parser.add_argument('--batch_size', type=int, default=160)  # original 72
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    #### model
    parser.add_argument('--depth',  type=int, default='50',
                         choices=[50,101,152])
    parser.add_argument('--num_features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--norm', type=bool, default=False)

    #### loss
    parser.add_argument('--loss_type', type=str, default='single')
    parser.add_argument('--sac', type=bool, default=True)
    parser.add_argument('--xen', type=bool, default=True)   
    parser.add_argument('--trip', type=bool, default=False)
    parser.add_argument('--pnsm_weight', type=bool, default=False)
    parser.add_argument('--pnsm_com', type=bool, default=False) # triplet combine weight and original
    parser.add_argument('--rank_loss', type=bool, default=False)  # triplet combine weight and original
    parser.add_argument('--quad', type=bool, default=False)

    parser.add_argument('--alpha', type=float, default= 1.0)
    parser.add_argument('--beta',  type=float, default= 1.0)
    parser.add_argument('--gamma', type=float, default= 1.0)
    parser.add_argument('--theta', type=float, default= 0.1)
    parser.add_argument('--margin1', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--margin2', type=float, default=0.42,
                        help="margin of the triplet2 loss, default: 0.1")
    parser.add_argument('--pnsm_margin', type=float, default=0.4,
                        help="margin of the triplet-weighted loss, default: 2.0")
    parser.add_argument('--pnsm_pos_radius_divide', type=float, default=2.0,
                        help="radius of the triplet-weighted loss, default: 2.0")
    parser.add_argument('--pnsm_neg_radius_divide', type=float, default=2.0,
                        help="radius of the triplet loss, default: 2.0")    
    #### optimizer
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of all parameters")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    #### training configs

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--frozen_sublayer', type=bool, default=False)

    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=310)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=1)
    #### metric learning
    parser.add_argument('--dist_metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    #### misc
    main(parser.parse_args())
