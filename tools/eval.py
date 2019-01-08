from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp

import numpy as np
import sys
import scipy.io as sio

sys.path.append("..")
import torch

th_version = torch.__version__

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import pdb
import yaml
from easydict import EasyDict as edict

from reid import datasets
from reid.models import model_creator
from reid.metric_learning import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.loss import merge_a_into_b
from reid.datasets.data import transforms as T
from reid.datasets.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint

from PIL import Image


class Processor2(Preprocessor):
    def __init__(self, dataset, root=None, transform=None):
        super(Processor2, self).__init__(dataset=dataset, root=root, transform=transform)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        # print(fpath)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, fpath


def check_cfg(args):
    if args.model == 'resnet_mgn' or args.model == 'resnet_mgn_lr':
        assert args.height == 384
        assert args.width == 128
    else:
        assert args.height == 256
        assert args.width == 128


# def get_data(name, split_id, data_dir, height, width, batch_size,
#              workers):
#     root = osp.join(data_dir, name)

#     dataset = datasets.create(name, root, split_id=split_id)

#     normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])

#     num_classes = dataset.num_trainval_ids
#     test_transformer = T.Compose([
#         T.RectScale(height, width),
#         T.ToTensor(),
#         normalizer,
#     ])
#     test_loader = DataLoader(
#         Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
#                      root=dataset.images_dir, transform=test_transformer),
#         batch_size=batch_size, num_workers=workers,
#         shuffle=False, pin_memory=True)

#     return dataset, num_classes, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    cfg_file = osp.join('.', 'cfgs', args.model, 'stage' + str(args.stage) + '.yml')
    if os.path.exists(cfg_file):
        print('cfg_file :', cfg_file)
        with open(cfg_file, 'r') as f:
            yaml_cfg = edict(yaml.load(f))
        merge_a_into_b(yaml_cfg, args.__dict__)
    else:
        print('cfg files not exists, using default params')
    check_cfg(args)

    root = osp.join(args.data_dir, args.dataset)
    dataset = datasets.create(args.dataset, root, split_id=args.split)
    num_classes = dataset.num_trainval_ids
    train_set = dataset.trainval
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(),
        normalizer,
    ])

    query_loader = Processor2(list(set(dataset.query)),
                              root=dataset.images_dir, transform=test_transformer)
    gallery_loader = Processor2(list(set(dataset.gallery)),
                                root=dataset.images_dir, transform=test_transformer)
    # trainset_loader = Processor2(train_set,
    #                              root=dataset.images_dir, transform=test_transformer)

    # Create model
    model = model_creator(args.model, args.depth, pretrained=True, num_features=args.num_features,
                          dropout=args.dropout, num_classes=num_classes, add_l3_softmax=False, test_stage=True)  # args.features

    # model = nn.DataParallel(model).cuda()
    # Load from checkpoint
    start_epoch = best_map = prior_best_map = 0
    if args.resume_path:
        ckpt_file = args.resume_path
    else:
        logs_base_dir = osp.join('.', 'logs', args.dataset, args.model)
        ckpt_dir = osp.join(logs_base_dir, 'stage' + str(args.stage))
        ckpt_file = osp.join(ckpt_dir, 'model_best.pth.tar')

    if not os.path.exists(ckpt_file):
        raise RuntimeError(ckpt_file + 'resume model doesnot exist!')
    else:
        print("resume model from", ckpt_file)
        checkpoint = load_checkpoint(ckpt_file)
        model.load_state_dict(checkpoint['state_dict'])
        curr_best_map = checkpoint['best_map']
        print("=> Start epoch {}  best map {:.1%}"
              .format(start_epoch, curr_best_map))


    model = nn.DataParallel(model).cuda()
    model.eval()


    count = 0
    for singleton in query_loader:
        # img, fname, pid, camid,fpath
        img = singleton[0]
        pid = singleton[2]
        camid = singleton[3]
        f_name = singleton[-1]
        print(f_name)
        file_name = osp.splitext(osp.basename(f_name))[0]
        # pdb.set_trace()
        # name.append(name)
        img_ori = img.permute(1, 2, 0).numpy()
        if th_version == '0.3.0.post4':
            img = Variable(img.unsqueeze(0), volatile=True)
        else:
            with torch.no_grad():
                img = Variable(img)

        output = model(img)
        pdb.set_trace()
        l3_attw, l4_attw, g_sm, g_rd, l2_side, l3_side, l4_side, net_rd = output
        l3_ca = l3_attw.cpu().data.numpy()
        l4_ca = l4_attw.cpu().data.numpy()
        fea = fea.cpu().data.numpy()
        l2_side = np.squeeze(l2_side.cpu().data.numpy())
        l3_side = np.squeeze(l3_side.cpu().data.numpy())
        l4_side = np.squeeze(l4_side.cpu().data.numpy())

        sio.savemat('./data_have_cll_multi/{}.mat'.format(file_name),
                    {'img': img_ori, 'name': f_name, 'fea': fea, 'pid': pid, 'camid': camid, 'l2_side': l2_side,
                     'l3_side': l3_side, 'l4_side': l4_side})
        count = count + 1
    print(count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='../dataset')
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['cuhk03', 'market1501', 'dukemtmc', 'market1501_2'])

    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--model', type=str, default='resnet_channel3',
                        choices=['resnet_mgn', 'resnet_mgn_lr', 'resnet_maxout2', 'resnet_reid', 
                                 'resnet_bypath','resnet_channel','resnet_channel2','resnet_channel3',])

    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")

    parser.add_argument('--batch_size', type=int, default=160)  # original 72
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    # model
    parser.add_argument('--depth', type=int, default='50',
                        choices=[50, 101, 152])
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
