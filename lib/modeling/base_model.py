"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `lib/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_test(model)
"""
import torch
import traceback
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from abc import ABCMeta
from abc import abstractmethod

from . import backbones
from .loss_aggregator import LossAggregator
from data.transform import get_transform
from data.collate_fn import CollateFn
from data.datasets.dataset import DataSet
from data.datasets.dataset_smplgait import DataSet_SMPLGait
from data.datasets.dataset_ccgait import DataSet_CCGait

import data.sampler as Samplers
from utils import Odict, mkdir, ddp_all_gather
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from utils import evaluation as eval_functions
from utils import NoOp
from utils import get_msg_mgr

__all__ = ['BaseModel']


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """
    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs, isTrain):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step."""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """Run a whole train schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """Run a whole test schedule."""
        raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.

    """

    def __init__(self, cfgs, training):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if training and self.engine_cfg['enable_float16']:
            self.Scaler = GradScaler()
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()

        self.msg_mgr.log_info(cfgs['data_cfg'])
        if training:
            self.train_loader, self.probe_seqs_num = self.get_loader(
                cfgs['data_cfg'], train=True)
        if not training or self.engine_cfg['with_test']:
            self.test_loader, self.probe_seqs_num = self.get_loader(
                cfgs['data_cfg'], train=False)

        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device(
            "cuda", self.device))

        if training:
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        self.train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

        if training:
            if cfgs['trainer_cfg']['fix_BN']:
                self.fix_BN()

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def build_network(self, model_cfg):
        if 'backbone_cfg' in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def get_loader(self, data_cfg, train=True):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        needsmpl = False
        needsmplsil = False
        needsmpleuler = False
        view_num = 0
        try:
            if data_cfg['dataset_root']['view_num'] is not None:
                view_num = data_cfg['dataset_root']['view_num']
        except:
            view_num = 0

        try:
            if data_cfg['dataset_root']['smpl_root'] is not None:
                print("needsmpl")
                needsmpl = True
        except:
            needsmpl = False
        try:
            if data_cfg['dataset_root']['smpl_sil_root'] is not None:
                print("needsmplprojectsil")
                needsmplsil = True
        except:
            needsmplsil = False
        try:
            if data_cfg['dataset_root']['smpl_euler_root'] is not None:
                print("needsmpleuler")
                needsmpleuler = True
        except:
            needsmpleuler = False
        
        if needsmplsil and view_num == 0:
            view_num = 1
        
        if train:
            if "DataSet_CCGait" in data_cfg['dataset_name']:
                dataset = DataSet_CCGait(data_cfg, train)
            elif needsmpl and not needsmplsil:
                self.msg_mgr.log_info("DataSet: DataSet_SMPLGait")
                dataset = DataSet_SMPLGait(data_cfg, train)
            else:
                dataset = DataSet(data_cfg, train)
        else:
            if "DataSet_CCGait" in data_cfg['test_dataset_name']:
                dataset = DataSet_CCGait(data_cfg, train)
            elif needsmpl and not needsmplsil and not needsmpleuler:
                self.msg_mgr.log_info("DataSet: DataSet_SMPLGait")
                dataset = DataSet_SMPLGait(data_cfg, train)
            else:
                dataset = DataSet(data_cfg, train)

        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        sampler = Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader, dataset.probe_seqs_num

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    def save_ckpt(self, iteration):
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            checkpoint = {
                'model': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iteration': iteration}
            torch.save(checkpoint,
                       osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']
        try:
            fine_tune = self.engine_cfg['fine_tune']
        except:
            fine_tune = False

        checkpoint = torch.load(save_name, map_location=torch.device(
            "cuda", self.device))
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training and not fine_tune:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()

    def inputs_pretreament(self, inputs, isTrain):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        try:
            twoSampleType = self.cfgs['trainer_cfg']['sampler']['two_types_sample']
        except:
            twoSampleType = False
        try:
            twoTypeFormat = self.cfgs['trainer_cfg']['sampler']['two_types_format']
        except:
            twoTypeFormat = None
        try:
            twoTStream = self.cfgs['trainer_cfg']['sampler']['two_time_stream']
        except:
            twoTStream = False


        try:
            withTsil = self.cfgs['trainer_cfg']['sampler']['with_Tsil']
        except:
            withTsil = False

        try:
            if isTrain:
                with_temporal_aug = self.cfgs['trainer_cfg']['sampler']['with_temporal_aug']
            else:
                with_temporal_aug = False
        except:
            with_temporal_aug = False
        
        try:
            view_num = self.cfgs['trainer_cfg']['sampler']['view_num']
        except:
            view_num = 0
        
        try:
            withSil = self.cfgs['trainer_cfg']['sampler']['with_sil']
        except:
            withSil = False

        try:
            # only smpl
            if 'smpl_root' in self.cfgs['data_cfg']['dataset_root'] \
                and ('smpl_sil_root' not in self.cfgs['data_cfg']['dataset_root']):

                seqs_smpls_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
                if not withSil:
                    seqs_batch = [seqs_smpls_batch[0]]
                    smpls_batch = [seqs_smpls_batch[1]]
                else:
                    seqs_batch = [seqs_smpls_batch[0]]
                    seqs_with_smpls_batch = [seqs_smpls_batch[1]]
                    smpls_batch = [seqs_smpls_batch[2]]

                trf_cfgs = self.engine_cfg['transform']
                seq_trfs = get_transform(trf_cfgs)

                requires_grad = bool(self.training)
                seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs, seqs_batch)]

                smpls = [np2var(np.asarray([fra for fra in smpl]), requires_grad=requires_grad).float()
                         for smpl in smpls_batch]


                if withSil:
                    with_smpls = [np2var(np.asarray([fra for fra in smpl]), requires_grad=requires_grad).float()
                         for smpl in seqs_with_smpls_batch]

                
                typeFlag = False
                if twoTStream:
                    typeFlag = True

                if twoSampleType and typeFlag:
                    temporal_batch = [seqs_smpls_batch[2]]
                    if twoTypeFormat == 'sil_smpl':
                        temps = [np2var(np.asarray([trf(fra) for fra in temp]), requires_grad=requires_grad).float()
                            for trf, temp in zip(seq_trfs, temporal_batch)]
                    else:
                        temps = [np2var(np.asarray([fra for fra in temp]), requires_grad=requires_grad).float()
                            for temp in temporal_batch]

                typs = typs_batch
                vies = vies_batch

                labs = list2var(labs_batch).long()

                if seqL_batch is not None:
                    seqL_batch = np2var(seqL_batch).int()
                seqL = seqL_batch

                if seqL is not None:
                    seqL_sum = int(seqL.sum().data.cpu().numpy())
                    ipts = [_[:, :seqL_sum] for _ in seqs]
                    sps = [_[:, :seqL_sum] for _ in smpls]
                    if twoSampleType and typeFlag:
                        tmps = [_[:, :seqL_sum] for _ in temps]
                    if withSil:
                        w_smpls = [_[:, :seqL_sum] for _ in with_smpls]
                else:
                    ipts = seqs
                    sps = smpls
                    if twoSampleType and typeFlag:
                        tmps = temps
                    if withSil:
                        w_smpls = with_smpls

                del seqs
                del smpls
                if twoSampleType and typeFlag:
                    del temps
                if withSil:
                    del with_smpls
                
                if not twoSampleType and not withSil:
                    return [ipts, sps], labs, typs, vies, seqL
                elif twoSampleType and not typeFlag and not withSil:
                    return [ipts, sps], labs, typs, vies, seqL
                elif withSil:
                    return [ipts, w_smpls, sps], labs, typs, vies, seqL
                else:
                    return [ipts, sps, tmps], labs, typs, vies, seqL
            # only smpl_sil
            elif 'smpl_sil_root' in self.cfgs['data_cfg']['dataset_root'] \
                and 'smpl_root' not in self.cfgs['data_cfg']['dataset_root'] :
                seqs_all_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
                trf_cfgs = self.engine_cfg['transform']
                seq_trfs = get_transform(trf_cfgs)

                if twoSampleType:
                    seqs_batch = [seqs_all_batch[0]]
                    temporal_batch = [seqs_all_batch[1]] 
                else:
                    seqs_batch = [seqs_all_batch[0]]
                    smpl_sil_batch = [seqs_all_batch[1]]

                requires_grad = bool(self.training)
                seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs, seqs_batch)]
                smpl_sil = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs, smpl_sil_batch)]

                if twoSampleType:
                    temps = [np2var(np.asarray([trf(fra) for fra in temp]), requires_grad=requires_grad).float()
                            for trf, temp in zip(seq_trfs, temporal_batch)]
            
                typs = typs_batch
                vies = vies_batch

                labs = list2var(labs_batch).long()

                if seqL_batch is not None:
                    seqL_batch = np2var(seqL_batch).int()
                seqL = seqL_batch

                if seqL is not None:
                    seqL_sum = int(seqL.sum().data.cpu().numpy())
                    ipts = [_[:, :seqL_sum] for _ in seqs]
                    s_ipts= [_[:, :seqL_sum] for _ in smpl_sil]
                    if twoSampleType:
                        tmps = [_[:, :seqL_sum] for _ in temps]
                else:
                    ipts = seqs
                    s_ipts = smpl_sil
                    if twoSampleType:
                        tmps = temps
            
                del seqs
                del smpl_sil
                if twoSampleType:
                    del temps
                return [ipts, s_ipts], labs, typs, vies, seqL
                if not twoSampleType:
                    return [ipts, s_ipts], labs, typs, vies, seqL
                else:
                    return [ipts, tmps], labs, typs, vies, seqL
            # smpl + smpl_sil
            elif 'smpl_sil_root' in self.cfgs['data_cfg']['dataset_root'] \
                and 'smpl_root' in self.cfgs['data_cfg']['dataset_root'] \
                and (view_num == 0 or view_num == 1):

                seqs_all_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
                trf_cfgs = self.engine_cfg['transform']
                trf_cfgs_smpl_sil = self.engine_cfg['transform_smpl_sil']
                seq_trfs = get_transform(trf_cfgs)
                seq_trfs_smpl_sil = get_transform(trf_cfgs_smpl_sil)
                
                np.set_printoptions(threshold=np.inf)
                seqs_batch = [seqs_all_batch[0]]
                smpls_batch = [seqs_all_batch[2]]
                smpl_sils_batch = [seqs_all_batch[1]]
                if withTsil:
                    seqs_tmp_batch = [seqs_all_batch[3]]
                if with_temporal_aug and not twoTStream:
                    smpls2_batch = [seqs_all_batch[3]]
                elif with_temporal_aug and twoTStream:
                    smpls2_batch = [seqs_all_batch[4]]
                if twoTStream:
                    tmp_smpl_sils_batch = [seqs_all_batch[3]]

                requires_grad = bool(self.training)
                seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs, seqs_batch)]
                if withTsil:
                    seqs_tmp = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                            for trf, seq in zip(seq_trfs, seqs_tmp_batch)]
                smpl_sils = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs_smpl_sil, smpl_sils_batch)]
                if twoTStream:
                    tmp_smpl_sils = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                            for trf, seq in zip(seq_trfs_smpl_sil, tmp_smpl_sils_batch)]

                smpls = [np2var(np.asarray([fra for fra in smpl]), requires_grad=requires_grad).float()
                         for smpl in smpls_batch]
                if with_temporal_aug:
                    smpls2 = [np2var(np.asarray([fra for fra in smpl]), requires_grad=requires_grad).float()
                            for smpl in smpls2_batch]

                typs = typs_batch
                vies = vies_batch

                labs = list2var(labs_batch).long()

                if seqL_batch is not None:
                    seqL_batch = np2var(seqL_batch).int()
                seqL = seqL_batch

                if seqL is not None:
                    seqL_sum = int(seqL.sum().data.cpu().numpy())
                    ipts = [_[:, :seqL_sum] for _ in seqs]
                    sps = [_[:, :seqL_sum] for _ in smpls]
                    spsils = [_[:, :seqL_sum] for _ in smpl_sils]
                    if twoTStream:
                        tmp_spsils = [_[:, :seqL_sum] for _ in tmp_smpl_sils]
                    if with_temporal_aug:
                        sps2 = [_[:, :seqL_sum] for _ in smpls2]
                    if withTsil:
                        ipts_t = [_[:, :seqL_sum] for _ in seqs_tmp]
                else:
                    ipts = seqs
                    sps = smpls
                    spsils = smpl_sils
                    if twoTStream:
                        tmp_spsils = tmp_smpl_sils
                    if with_temporal_aug:
                        sps2 = smpls2
                    if withTsil:
                        ipts_t = seqs_tmp
                
                del seqs
                del smpls
                del smpl_sils
                if twoTStream:
                    del tmp_smpl_sils
                if with_temporal_aug:
                    del smpls2
                if withTsil:
                    del seqs_tmp

                if twoTStream and not with_temporal_aug:
                    return [ipts, spsils, sps, tmp_spsils], labs, typs, vies, seqL
                if withTsil:
                    return [ipts, spsils, sps, ipts_t], labs, typs, vies, seqL
                if with_temporal_aug and not twoTStream:
                    return [ipts, spsils, sps, sps2], labs, typs, vies, seqL
                if with_temporal_aug and twoTStream:
                    return [ipts, spsils, sps, tmp_spsils, sps2], labs, typs, vies, seqL
                return [ipts, spsils, sps], labs, typs, vies, seqL
            # smpl + smpl_sil*2
            elif 'smpl_sil_root' in self.cfgs['data_cfg']['dataset_root'] \
                and 'smpl_root' in self.cfgs['data_cfg']['dataset_root'] \
                and view_num == 2 :
                seqs_all_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
                trf_cfgs = self.engine_cfg['transform']
                trf_cfgs_smpl_sil = self.engine_cfg['transform_smpl_sil']
                seq_trfs = get_transform(trf_cfgs)
                seq_trfs_smpl_sil = get_transform(trf_cfgs_smpl_sil)
                
                np.set_printoptions(threshold=np.inf)
                # print(len(seqs_all_batch))
                seqs_batch = [seqs_all_batch[0]]
                smpls_batch = [seqs_all_batch[3]]
                smpl_sils_batch = [seqs_all_batch[1]]
                smpl_sils_2_batch = [seqs_all_batch[2]]

                requires_grad = bool(self.training)
                seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs, seqs_batch)]
                smpl_sils = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs_smpl_sil, smpl_sils_batch)]
                smpl_sils_2 = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs_smpl_sil, smpl_sils_2_batch)]

                smpls = [np2var(np.asarray([fra for fra in smpl]), requires_grad=requires_grad).float()
                         for smpl in smpls_batch]

                typs = typs_batch
                vies = vies_batch

                labs = list2var(labs_batch).long()

                if seqL_batch is not None:
                    seqL_batch = np2var(seqL_batch).int()
                seqL = seqL_batch

                if seqL is not None:
                    seqL_sum = int(seqL.sum().data.cpu().numpy())
                    ipts = [_[:, :seqL_sum] for _ in seqs]
                    sps = [_[:, :seqL_sum] for _ in smpls]
                    spsils = [_[:, :seqL_sum] for _ in smpl_sils]
                    spsils_2 = [_[:, :seqL_sum] for _ in smpl_sils_2]
                else:
                    ipts = seqs
                    sps = smpls
                    spsils = smpl_sils
                    spsils_2 = smpl_sils_2
                
                del seqs
                del smpls
                del smpl_sils
                del smpl_sils_2

                return [ipts, spsils, spsils_2, sps], labs, typs, vies, seqL

            elif 'smpl_sil_root' in self.cfgs['data_cfg']['dataset_root'] \
                and 'smpl_root' in self.cfgs['data_cfg']['dataset_root'] \
                and view_num == 3 :
                seqs_all_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
                trf_cfgs = self.engine_cfg['transform']
                trf_cfgs_smpl_sil = self.engine_cfg['transform_smpl_sil']
                seq_trfs = get_transform(trf_cfgs)
                seq_trfs_smpl_sil = get_transform(trf_cfgs_smpl_sil)
                
                np.set_printoptions(threshold=np.inf)
                # print(len(seqs_all_batch))
                seqs_batch = [seqs_all_batch[0]]
                smpls_batch = [seqs_all_batch[4]]
                smpl_sils_batch = [seqs_all_batch[1]]
                smpl_sils_2_batch = [seqs_all_batch[2]]
                smpl_sils_3_batch = [seqs_all_batch[3]]

                requires_grad = bool(self.training)
                seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs, seqs_batch)]
                smpl_sils = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs_smpl_sil, smpl_sils_batch)]
                smpl_sils_2 = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs_smpl_sil, smpl_sils_2_batch)]
                smpl_sils_3 = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                        for trf, seq in zip(seq_trfs_smpl_sil, smpl_sils_3_batch)]
 
                smpls = [np2var(np.asarray([fra for fra in smpl]), requires_grad=requires_grad).float()
                         for smpl in smpls_batch]

                typs = typs_batch
                vies = vies_batch

                labs = list2var(labs_batch).long()

                if seqL_batch is not None:
                    seqL_batch = np2var(seqL_batch).int()
                seqL = seqL_batch

                if seqL is not None:
                    seqL_sum = int(seqL.sum().data.cpu().numpy())
                    ipts = [_[:, :seqL_sum] for _ in seqs]
                    sps = [_[:, :seqL_sum] for _ in smpls]
                    spsils = [_[:, :seqL_sum] for _ in smpl_sils]
                    spsils_2 = [_[:, :seqL_sum] for _ in smpl_sils_2]
                    spsils_3 = [_[:, :seqL_sum] for _ in smpl_sils_3]
                else:
                    ipts = seqs
                    sps = smpls
                    spsils = smpl_sils
                    spsils_2 = smpl_sils_2
                    spsils_3 = smpl_sils_3
                
                del seqs
                del smpls
                del smpl_sils
                del smpl_sils_2
                del smpl_sils_3

                return [ipts, spsils, spsils_2, spsils_3, sps], labs, typs, vies, seqL

            if self.cfgs['data_cfg']['dataset_root']['smpl_root'] is not None:
                    flag = True
        except (Exception, BaseException) as e:
            # print(e)
            exstr = traceback.format_exc()
            # print(exstr)
            seqs_all_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
            trf_cfgs = self.engine_cfg['transform']
            seq_trfs = get_transform(trf_cfgs)

            if twoSampleType:
                seqs_batch = [seqs_all_batch[0]]
                temporal_batch = [seqs_all_batch[1]] 
            else:
                seqs_batch = [seqs_all_batch[0]]

            requires_grad = bool(self.training)
            seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                    for trf, seq in zip(seq_trfs, seqs_batch)]

            if twoSampleType:
                temps = [np2var(np.asarray([trf(fra) for fra in temp]), requires_grad=requires_grad).float()
                         for trf, temp in zip(seq_trfs, temporal_batch)]
            
            typs = typs_batch
            vies = vies_batch

            labs = list2var(labs_batch).long()

            if seqL_batch is not None:
                seqL_batch = np2var(seqL_batch).int()
            seqL = seqL_batch

            if seqL is not None:
                seqL_sum = int(seqL.sum().data.cpu().numpy())
                ipts = [_[:, :seqL_sum] for _ in seqs]
                if twoSampleType:
                    tmps = [_[:, :seqL_sum] for _ in temps]
            else:
                ipts = seqs
                if twoSampleType:
                    tmps = temps
            
            del seqs
            if twoSampleType:
                del temps

            if not twoSampleType:
                return ipts, labs, typs, vies, seqL
            else:
                return [ipts, tmps], labs, typs, vies, seqL

    def train_step(self, loss_sum, model) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                    scale, self.Scaler.get_scale()))
                return False
        else:
            loss_sum.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name)
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()
        return True

    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        # visualization
        # counter = 0
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs, False)
            # print("ipts.shape:{}".format(len(ipts[0])))
            with autocast(enabled=self.engine_cfg['enable_float16']):
                # retval = self.forward(ipts, counter)
                retval = self.forward(ipts)
                # counter += 1
                inference_feat = retval['inference_feat']
                for k, v in inference_feat.items():
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict

    @ staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        for inputs in model.train_loader:
            ipts = model.inputs_pretreament(inputs, True)
            # print("ipts.shape:{}".format(len(ipts[0])))
            with autocast(enabled=model.engine_cfg['enable_float16']):
                retval = model(ipts)
                training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                del retval
            loss_sum, loss_info = model.loss_aggregator(training_feat)
            ok = model.train_step(loss_sum, model)
            if not ok:
                continue

            visual_summary.update(loss_info)
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']

            model.msg_mgr.train_step(loss_info, visual_summary)
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # save the checkpoint
                model.save_ckpt(model.iteration)

                # run test if with_test = true
                if model.engine_cfg['with_test']:
                    model.msg_mgr.log_info("Running test...")
                    model.eval()
                    result_dict = BaseModel.run_test(model)
                    model.train()
                    # model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break

    @ staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""

        rank = torch.distributed.get_rank()
        with torch.no_grad():
            info_dict = model.inference(rank)
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            # CCGait
            info_dict.update({
                'labels': label_list, 'types': types_list})
            
            # Gait3d
            # views_list = loader.dataset.views_list
            # info_dict.update({
            #     'labels': label_list, 'types': types_list, 'views': views_list})

            

            if 'eval_func' in model.cfgs["evaluator_cfg"].keys():
                eval_func = model.cfgs['evaluator_cfg']["eval_func"]
            else:
                eval_func = 'identification'
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, model.cfgs["evaluator_cfg"], ['metric'])
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']

            if model.engine_cfg['eval_func'] in ["evaluation_Gait3D", "evaluation_GREW", "evaluation_LTGait"]:
                return eval_func(info_dict, model.cfgs, model.probe_seqs_num, **valid_args)
            else:
                return eval_func(info_dict, dataset_name, **valid_args)
