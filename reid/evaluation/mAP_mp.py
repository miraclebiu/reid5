from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

import pdb
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import time

from ..utils import to_torch


def accuracy(output, target, topk=(1,)):
    output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret



def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    start = time.time()
    indices = np.argsort(distmat, axis=1)
    end = time.time()
    print('sort_time:', end-start)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    start = time.time()
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    end = time.time()
    print('ap_time:', end-start)
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)



def cal_sort(distmat):
    return np.argsort(distmat)

def cal_aps_mp(distmat,match,gallery_id,query_id,gallery_cam,query_cam):
    # pdb.set_trace()
    valid = ((gallery_id != query_id) |
             (gallery_cam != query_cam))
    y_true = match[valid]
    # print(distmat)
    y_score = - distmat[valid]
    if not np.any(y_true): return 0
    return average_precision_score(y_true, y_score)

def run_func(args):
    return cal_aps_mp(args[0],args[1],args[2],args[3],args[4],args[5])


def mean_ap_mp(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):# multiprocess
    worker_num = 16
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)

    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    print('cal_sort')
    start = time.time()
    pool = ThreadPool(processes=worker_num)
    each_dist = [distmat[i] for i in range(m)]
    results = pool.map(cal_sort,each_dist)
    pool.close()
    pool.join()
    indices = np.vstack(results)
    end = time.time()
    print('sort_time:',end-start)

    print('cal_map')
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    params=[]
    start = time.time()

    for i in range(m):
        gallery_ids_map = gallery_ids[indices[i]]
        gallery_cam_map = gallery_cams[indices[i]]
        distmat_map = distmat[i][indices[i]]
        query_id_map = query_ids[i]
        query_cam_map = query_cams[i]
        match_map = matches[i]
        params.append([distmat_map,match_map,gallery_ids_map,query_id_map,gallery_cam_map,query_cam_map])

    pool = ThreadPool(processes=worker_num)

    results = pool.map(run_func, params)
    pool.close()
    pool.join()
    end = time.time()
    print('ap_time:',end-start)
    if len(results) == 0:
        raise RuntimeError("No valid query")
    return np.mean(results)

def main():
    alll = np.load('../../mean_ap.npz')
    distmat=alll['distmat'] 
    query_ids=alll['query_ids']
    gallery_ids = alll['gallery_ids']
    query_cams = alll['query_cams']
    gallery_cams = alll['gallery_cams']
    print('load Done!')
    map1 = mean_ap_mp(distmat,query_ids,gallery_ids, query_cams, gallery_cams)
    map2 = mean_ap(distmat,query_ids,gallery_ids, query_cams, gallery_cams)
    print('multi-process:',map1)
    print('single-process:', map2)


if __name__ == '__main__':
    main()