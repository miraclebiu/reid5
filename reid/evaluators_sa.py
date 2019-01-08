from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from torch.nn import functional as F
import pdb

from .evaluation import extract_cnn_feature, cmc, mean_ap_mp, AverageMeter
from .utils import to_numpy

import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing

def selfattention(q, k, v, scale):
    attention = torch.bmm(q, k.transpose(1, 2))
    attention = attention * scale
    attention = F.softmax(attention)
    context = torch.bmm(attention, v)
    return context, attention


def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x,axis=0)

def cal_sort(distmat):
    return np.argsort(distmat)

def extract_features(model, data_loader, print_freq=1, metric=None,if_flip=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels




def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        qfea = torch.cat(list(features.values()))
        qfea = qfea.view(n, -1)
        dist = torch.pow(qfea, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(qfea, qfea.t())
        return dist

    qfea = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    gfea = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = qfea.size(0), gfea.size(0)
    # pdb.set_trace()
    qfea = qfea.view(m, -1)
    gfea = gfea.view(n, -1)

    # q2g = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # q2g.addmm_(1, -2, x, y.t())
    # q2g = q2g.clamp(min=1e-12).sqrt()
    
    q2g = torch.mm(qfea, gfea.t())



    topk_num = 50
    topk_dist, topk_ind = q2g.topk(topk_num)

    temperature = 0.1
    topk_dist = topk_dist / temperature

    weight = F.softmax(topk_dist,1)

    m = qfea.size(0)
    sft_qfea = []
    for i in range(m):
        temp_fea = torch.index_select(gfea,0,topk_ind[i])
        temp_weight = weight[i].unsqueeze(1)
        new_fea = temp_weight * temp_fea
        sft_qfea.append(new_fea.sum(0))
    # pdb.set_trace()
    qfea2 = torch.stack(sft_qfea)
    qfea2 = qfea + qfea2
    qfea2 = F.normalize(qfea2)

    q2g2 = torch.mm(qfea2, gfea.t())







    # g2g = torch.mm(y, y.t())
    # q2g = torch.mm(q2g,g2g)
    return q2g2


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10),eval_cmc=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    distmat2 = to_numpy(distmat)
    mAP = mean_ap_mp(-distmat2, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    if eval_cmc:
        cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

        print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))
        return cmc_scores['allshots'][0]
    # Use the allshots cmc top-1 score for validation criterion
    return mAP


class Evaluator_sa(object):
    def __init__(self, model):
        super(Evaluator_sa, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None, eval_cmc=False):
        features, _ = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery, eval_cmc=eval_cmc)
