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

from collections import OrderedDict

import time

from reid import datasets
from reid.models import model_creator
from reid.metric_learning import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.evaluation import AverageMeter
from reid.loss import merge_a_into_b
from reid.datasets import get_data
from reid.datasets.data import transforms as T
from reid.datasets.data.preprocessor import Preprocessor
from reid.utils.serialization import load_checkpoint

from PIL import Image

import shutil
import zipfile


def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    zipf.close()


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
    


    # Create model
    model = model_creator(args.model, args.depth, pretrained=True, num_features=args.num_features,
                          dropout=args.dropout, num_classes=num_classes, add_l3_softmax=False, test_stage=True)  # args.features

    # model = nn.DataParallel(model).cuda()
    # Load from checkpoint
    start_epoch = best_map = prior_best_map = 0
    if args.resume_path:
        ckpt_file = args.resume_path
    else:
        logs_base_dir = osp.join('.', 'logs', 'market1501', args.model)
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
        fname = singleton[1]
        file_name = osp.splitext(fname)[0]
        with torch.no_grad():
            img_forward = Variable(img.unsqueeze(0))     

        img_ori = img.permute(1, 2, 0).numpy()

        l3_ca, l4_ca, _, _, l2_side, l3_side, l4_side, net_rd =  model(img_forward)
        # pdb.set_trace()


        l3_ca = l3_ca.cpu().data.numpy()
        l4_ca = l4_ca.cpu().data.numpy()
        fea = net_rd.cpu().data.numpy()
        l2_side = np.squeeze(l2_side.cpu().data.numpy())
        l3_side = np.squeeze(l3_side.cpu().data.numpy())
        l4_side = np.squeeze(l4_side.cpu().data.numpy())
        # pdb.set_trace()
        sio.savemat('./result_visual/{}.mat'.format(file_name),
                    {'img': img_ori, 'name': fname, 'fea': fea, 'pid': pid, 'camid': camid, 'l2_side': l2_side,
                     'l3_side': l3_side, 'l4_side': l4_side, 'l3_ca': l3_ca, 'l4_ca': l4_ca})
        print(count)
        count = count + 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='../dataset')
    parser.add_argument('--dataset', type=str, default='market1501_2',
                        choices=['cuhk03', 'market1501', 'dukemtmc', 'market1501_2'])

    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--model', type=str, default='resnet_channel3')

    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")

    parser.add_argument('--batch_size', type=int, default=128)  # original 72
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
