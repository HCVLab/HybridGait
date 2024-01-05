# -*- coding: utf-8 -*-
"""
   File Name：     rank
   Author :       jinkai Zheng
   date：          2022/1/17
   E-mail:        zhengjinkai3@qq.com
"""

import numpy as np


def evaluate_rank(distmat, p_lbls, g_lbls, probe_cams, gallery_cams, probe_time_seqs, gallery_time_seqs, max_rank=50):

    num_p, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    c = open("res4.txt", "w+")

    np.set_printoptions(threshold=np.inf)

    print("indices.shape:{}".format(len(indices)))
    print("indices.shape:{}".format(len(indices[0])))
    print("g_lbls.shape:{}".format(len(g_lbls)))
    print("g_lbls.shape:{}".format(len(g_lbls[0])))
    print("g_lbls:{}".format(g_lbls), file=c)
    print("p_lbls.shape:{}".format(len(p_lbls)))
    print("p_lbls:{}".format(p_lbls), file=c)
    print("indices:{}".format(indices), file=c)
    print("probe_cams:{}".format(probe_cams), file=c)
    print("gallery_cams:{}".format(gallery_cams), file=c)
    print("probe_time_seqs:{}".format(probe_time_seqs), file=c)
    print("gallery_time_seqs:{}".format(gallery_time_seqs), file=c)
    

    matches = (g_lbls[indices] == p_lbls[:, np.newaxis]).astype(np.int32)
    print("p_lbls:{}".format(p_lbls), file=c)
    print("matches:{}".format(matches), file=c)

    # compute cmc curve for each probe
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_p = 0.  # number of valid probe

    for p_idx in range(num_p):
        # compute cmc curve
        raw_cmc = matches[p_idx]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when probe identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)    # 返回坐标，此处raw_cmc为一维矩阵，所以返回相当于index
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_p > 0, 'Error: all probe identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_p

    return all_cmc, all_AP, all_INP




def evaluate_rank_LTGait(distmat, p_lbls, g_lbls, probe_cams, gallery_cams, max_rank=50):

    num_p, num_g = distmat.shape
    print("num_p:{}".format(num_p))
    print("num_g:{}".format(num_g))

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    c = open("res4.txt", "w+")

    np.set_printoptions(threshold=np.inf)

    print("indices.shape:{}".format(len(indices)))
    # print("indices.shape:{}".format(len(indices[0])))
    print("g_lbls.shape:{}".format(len(g_lbls)))
    print("g_lbls.shape:{}".format(len(g_lbls[0])))
    print("g_lbls:{}".format(g_lbls), file=c)
    print("p_lbls.shape:{}".format(len(p_lbls)))
    print("p_lbls:{}".format(p_lbls), file=c)
    print("indices:{}".format(indices), file=c)
    print("probe_cams:{}".format(probe_cams), file=c)
    print("gallery_cams:{}".format(gallery_cams), file=c)
    

    matches = (g_lbls[indices] == p_lbls[:, np.newaxis]).astype(np.int32)
    print("p_lbls:{}".format(p_lbls), file=c)
    print("matches:{}".format(matches), file=c)

    # compute cmc curve for each probe
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_p = 0.  # number of valid probe

    for p_idx in range(num_p):
        # compute cmc curve
        raw_cmc = matches[p_idx]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when probe identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)    # 返回坐标，此处raw_cmc为一维矩阵，所以返回相当于index
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_p > 0, 'Error: all probe identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_p

    return all_cmc, all_AP, all_INP

