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


def check_cfg(args):
    if args.model == 'resnet_mgn' or args.model == 'resnet_mgn_lr':
        assert args.height == 384
        assert args.width == 128
    else:
        assert args.height == 256
        assert args.width == 128



def extract_features(model, data_loader, print_freq=1, metric=None,if_flip=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    l3_cas = OrderedDict()
    l4_cas = OrderedDict()
    l2_sides = OrderedDict()
    l3_sides = OrderedDict()
    l4_sides = OrderedDict()
    files = list()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            imgs = Variable(imgs)
        # outputs =  model(imgs)
        outputs =  model(imgs)
        
        l3_ca, l4_ca, _, _, l2_side, l3_side, l4_side, net_rd = outputs
        net_rd = net_rd.data.cpu()
        for fname, output, pid in zip(fnames, net_rd, pids):

            features[fname] = output
            labels[fname] = pid
            files.append(fname)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, labels, files

def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist




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
    print(args.dataset)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, 16, 8, 2,
                 True)

    # Create model
    model = model_creator(args.model, args.depth, pretrained=True, num_features=args.num_features,
                          dropout=args.dropout, num_classes=num_classes, add_l3_softmax=False, test_stage=True)  # args.features


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

    features, labels, fnames  = extract_features(model, test_loader)
    distmat = pairwise_distance(features, dataset.query, dataset.gallery)
    num_query = len(dataset.query)
    query_files = list(map(lambda x: x[0],dataset.query))
    gallery_files = list(map(lambda x: x[0],dataset.gallery))

    retrieve_imgs = {}
    show_index = 15

    img_files_dir = osp.join(args.data_dir,args.dataset,'images')
    res_dir = osp.join('.','result_retrieval')
    os.mkdir(res_dir)

    for i in range(num_query):
        query_img = query_files[i]
        dist_single = distmat[i]
        sort_dist, sort_index = dist_single.sort()
        top_index = sort_index[:show_index]
        top_index = top_index.numpy().tolist()
        top_imgs = [gallery_files[i] for i in top_index]
        # pdb.set_trace()
        save_dir = query_img.split('.')[0]
        save_dir = osp.join(res_dir , save_dir)

        os.mkdir(save_dir)
        query_save_path = osp.join(save_dir,'query_img.jpg')
        query_ori_path = osp.join(img_files_dir,query_img)
        shutil.copy(query_ori_path,query_save_path)
        count = 0
        for each_img in top_imgs:

            img_base_name = 'rank' + str(count) +'_'+ osp.splitext(each_img)[0] + '.jpg'
            each_save_path = osp.join(save_dir,img_base_name)
            each_ori_path = osp.join(img_files_dir, each_img)
            shutil.copy(each_ori_path, each_save_path)
            count +=1
        print(query_img)


    source_dir = res_dir
    filename = 'retrieval_results.zip'
    make_zip(source_dir,filename)







    #     l3_attw, l4_attw, g_sm, g_rd, l2_side, l3_side, l4_side, net_rd = output
    #     l3_ca = l3_attw.cpu().data.numpy()
    #     l4_ca = l4_attw.cpu().data.numpy()
    #     fea = net_rd.cpu().data.numpy()
    #     l2_side = np.squeeze(l2_side.cpu().data.numpy())
    #     l3_side = np.squeeze(l3_side.cpu().data.numpy())
    #     l4_side = np.squeeze(l4_side.cpu().data.numpy())

    #     sio.savemat('./data_have_cll_multi/{}.mat'.format(file_name),
    #                 {'img': img_ori, 'name': f_name, 'fea': fea, 'pid': pid, 'camid': camid, 'l2_side': l2_side,
    #                  'l3_side': l3_side, 'l4_side': l4_side, 'l3_ca': l3_ca, 'l4_ca': l4_ca})
    #     count = count + 1
    # print(count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='../dataset')
    parser.add_argument('--dataset', type=str, default='market1501_2',
                        choices=['cuhk03', 'market1501', 'dukemtmc', 'market1501_2'])

    parser.add_argument('--stage', type=int, default=3)
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
