import math
import random
import numpy as np
from utils import get_msg_mgr


class CollateFn(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        sample_type = sample_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]
        try:
            self.with_temporal_aug = sample_config['with_temporal_aug']
        except:
            self.with_temporal_aug = False

        try:
            self.view_num = sample_config['view_num']
        except:
            self.view_num = 0

        try:
            self.twoType = sample_config['two_types_sample']
        except:
            self.twoType = False
        try:
            self.twoTypeFormat = sample_config['two_types_format']
        except:
            self.twoTypeFormat = False
        try:
            self.twoTStream = sample_config['two_time_stream']
        except:
            self.twoTStream = False

        try:
            self.withSil = sample_config['with_sil']
        except:
            self.withSil = False

        try:
            self.withTsil = sample_config['with_Tsil']
            # print(sample_config['with_Tsil'])
        except:
            self.withTsil = False
            # print(sample_config['with_Tsil'])

        try:
            self.plus_smpl = sample_config['plus_smpl']
        except:
            self.plus_smpl = False
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError
        if self.ordered not in ['ordered', 'unordered']:
            raise ValueError
        self.ordered = sample_type[1] == 'ordered'

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed']

        # unfixed cases
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max']
            self.frames_num_min = sample_config['frames_num_min']

        if self.twoType or (self.sampler != 'all' and self.ordered):
            self.frames_skip_num = sample_config['frames_skip_num']

        self.frames_all_limit = -1
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']

    def __call__(self, batch):
        batch_size = len(batch)
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now.
        # We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        if self.plus_smpl:
            feature_num = len(batch[0][0][0])
        else:
            feature_num = len(batch[0][0])
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []
        
        if self.twoType:
            if self.twoTypeFormat == 'smpl_smpl_sil' \
                or self.twoTypeFormat == 'sil_smpl_sil' \
                or self.twoTypeFormat == 'sil_smpl' \
                or self.twoTypeFormat == 'smpl_sil_smpl' \
                or self.twoTypeFormat == 'smpl_smpl_euler':
                # or (self.twoTypeFormat == 'smpl' and self.withSil):
                feature_num = 3
            # if self.plus_smpl:
            #     feature_num += 1
                if self.view_num > 0:
                    feature_num = feature_num - 1 + self.view_num
            elif self.twoTypeFormat == 'smpl_sil_smpl_smpl_sil':
                feature_num = 4
                if self.view_num > 0:
                    feature_num = feature_num - 1 + self.view_num
            else:
                feature_num = 2

            if self.withTsil:
                feature_num += 1

        for bt in batch:
            if self.plus_smpl:
                seqs_batch.append(bt[0][0])
            else:
                seqs_batch.append(bt[0])
            labs_batch.append(self.label_set.index(bt[1][0]))
            typs_batch.append(bt[1][1])
            vies_batch.append(bt[1][2])

        global count
        count = 0

        def sample_frames(seqs):
            global count
            if not self.with_temporal_aug:
                sampled_fras = [[] for i in range(feature_num)]
            else:
                sampled_fras = [[] for i in range(feature_num+1)]
            seq_len = len(seqs[0])
            indices = list(range(seq_len))
            t_indices = list(range(seq_len))
            if self.with_temporal_aug:
                t_indices_2 = list(range(seq_len))

            if self.sampler in ['fixed', 'unfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))

                if self.ordered or self.twoType:
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                    if self.twoType:
                        t_indices = indices
                
                if self.with_temporal_aug and (self.ordered or self.twoType):
                    seq_len = len(seqs[0])
                    indices = list(range(seq_len))
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it
                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                    if self.twoType:
                        t_indices_2 = indices

                if not self.ordered:
                    seq_len = len(seqs[0])
                    indices = list(range(seq_len))

                    replace = seq_len < frames_num

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))
                                                
                    count += 1
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)

            for i in range(feature_num):
                if  (self.twoType and self.withTsil and not self.twoTStream and i != feature_num-2) or \
                    (self.twoType and not self.withTsil and not self.twoTStream and i != feature_num-1) or \
                    (not self.twoType) or \
                    (self.twoType and not self.withTsil and self.twoTStream and self.twoTypeFormat == 'smpl_sil_smpl_smpl_sil' and i != feature_num-1 and i != feature_num-2) or \
                    (self.twoType and not self.withTsil and self.twoTStream and i == 0 and self.twoTypeFormat != 'smpl_sil_smpl_smpl_sil'):

                    typeformatlist = ['smpl_sil_smpl', 'smpl_sil_smpl_smpl_sil']
                    if self.twoTypeFormat in typeformatlist and (self.view_num==0 or self.view_num==1) and i != 0:
                        index = 2
                    elif self.twoTypeFormat in typeformatlist and (self.view_num==0 or self.view_num==1) and i == 0:
                        index = 0
                    elif self.twoTypeFormat in typeformatlist and self.view_num==2 and i == 1:
                        index = 2
                    elif self.twoTypeFormat in typeformatlist and self.view_num==2 and i == 2:
                        index = 3
                    elif self.twoTypeFormat in typeformatlist and self.view_num==2 and i == 0:
                        index = 0  
                    elif self.twoTypeFormat in typeformatlist and self.view_num==3 and i == 1:
                        index = 2
                    elif self.twoTypeFormat in typeformatlist and self.view_num==3 and i == 2:
                        index = 3
                    elif self.twoTypeFormat in typeformatlist and self.view_num==3 and i == 3:
                        index = 4  
                    elif self.twoTypeFormat in typeformatlist and self.view_num==3 and i == 0:
                        index = 0 
                    else:
                        index = i
                    for j in indices[:self.frames_all_limit] if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit else indices:
                        sampled_fras[i].append(seqs[index][j])
                else:
                    # ordered
                    if self.twoTypeFormat == 'sil' or self.twoTypeFormat == 'sil_smpl':
                        index = 0
                    elif self.twoTypeFormat == 'smpl_smpl_sil' or self.twoTypeFormat == 'smpl_smpl_euler':
                        index = 2
                    elif self.twoTypeFormat == 'smpl' or self.twoTypeFormat == 'smpl_sil' or self.twoTypeFormat == 'smpl_sil_smpl' or self.twoTypeFormat == 'smpl_sil_smpl_smpl_sil':
                        index = 1
                    else:
                        raise KeyError ()
                        
                    if self.twoTStream and i == 1:
                        index = 1
                    if self.twoTypeFormat == 'smpl_sil_smpl' and self.twoTStream and i == 1:
                        index = 1
                    if self.twoTypeFormat == 'smpl_sil_smpl' and self.twoTStream and i == 2:
                        index = 2
                    if self.twoTStream and self.twoTypeFormat == 'smpl_sil_smpl_smpl_sil' and i == 2:
                        index = 1
                    if self.twoTStream and self.twoTypeFormat == 'smpl_sil_smpl_smpl_sil' and i == 3:
                        index = 2

                    if self.withTsil and i == feature_num-1:
                        index = 0

                    for j in t_indices[:self.frames_all_limit] if self.frames_all_limit > -1 and len(t_indices) > self.frames_all_limit else t_indices:

                        sampled_fras[i].append(seqs[index][j])

            if self.with_temporal_aug:
                index = 1
                for j in t_indices_2[:self.frames_all_limit] if self.frames_all_limit > -1 and len(t_indices_2) > self.frames_all_limit else t_indices_2:
                        sampled_fras[i+1].append(seqs[index][j])
            # print(len(sampled_fras))
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f]
        batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        if self.sampler == "fixed":
            # print(len(fras_batch[0]))
            # print("feature+1:{}".format(feature_num+1))
            if not self.with_temporal_aug:
                fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [f, b]
            if self.with_temporal_aug:
                fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num+1)] 
        else:
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # [1, p]

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0)
            if not self.with_temporal_aug:
                fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]
            if self.with_temporal_aug:
                fras_batch = [[my_cat(k)] for k in range(feature_num+1)]

            batch[-1] = np.asarray(seqL_batch)

        batch[0] = fras_batch
        return batch
