import os
import time
import numpy as np
import argparse
import datetime
import wandb

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

import timm.optim.optim_factory as optim_factory
from timm.utils import CheckpointSaver

from utils import create_BrainEncoder, save_config, adjust_learning_rate, update_config, build_loss, build_metrics
from config import Config_BrainEncoder_Bases
from dataset import NSDDataset
from trainer import NativeScalerWithGradNormCount as NativeScaler

class wandb_logger:
    def __init__(self, config):
        wandb.init(
                    project=config.project,
                    anonymous="allow",
                    group='bases',
                    config=config,
                    reinit=True)

        self.config = config
        self.step = None
    
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def finish(self):
        wandb.finish(quiet=True)

def get_args_parser():
    parser = argparse.ArgumentParser('iMIND Dual Decoding', add_help=False)
    
    # Project setting
    parser.add_argument('--seed', type=str)

    # Model Parameters
    parser.add_argument('--num_ca_heads', type=int)
    parser.add_argument('--dim_subject', type=int)
    parser.add_argument('--dim_object', type=int)
    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--bases_loss_w', type=float)   
    parser.add_argument('--subj_loss_w', type=float)   
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
    parser.add_argument("--local-rank", type=int, default=0)
                        
    return parser


def train_one_epoch(_model, _loader, _optimizer, _device, _epoch, 
                    _loss_fn, _eval_metrics, _loss_scaler, _logger, _config=None, _start_time=None):
    
    _model.train()
    _loss_res = torch.zeros(len(_loss_fn) + 1)
    accum_iter = _config.accum_iter
        
    for _bid, _data in enumerate(tqdm(_loader)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if _bid % accum_iter == 0:
            adjust_learning_rate(_optimizer, _bid / len(_loader) + _epoch, _config)
            
        _data = {k: d.to(_device) for k, d in _data.items()}
        
        with torch.amp.autocast('cuda'):
            _feat_obj, _pred_obj, _pred_subj = _model(**_data)
            
        _, _loss = _loss_fn(pred_obj=_pred_obj, 
                            pred_subj=_pred_subj,
                            b_obj=_model.module.os_extractor.b_subj.weight,
                            b_subj=_model.module.os_extractor.b_obj.weight,
                            # b_obj=_model.os_extractor.b_subj.weight,
                            # b_subj=_model.os_extractor.b_obj.weight, 
                            fmri_obj = _pred_obj,
                            **_data)
        _optimizer.zero_grad()
        
            
        if _loss.isnan() or _loss.isinf():
            raise ValueError(f"Loss is {_}, stopping training at step {_bid} epoch {_epoch}")
        
        _loss_scaler(_loss, _optimizer, parameters=_model.parameters(), clip_grad=_config.clip_grad)
        
        _eval_metrics.update(pred_obj=_pred_obj, pred_subj=_pred_subj, **_data)
        _loss_res += torch.hstack([_, _loss]).cpu()
    
    
    _loss_res /= len(_loader)
    _metrics_res = _eval_metrics.compute()
    _eval_metrics.reset()
        
        
    if _logger is not None:        
        _logger.log('lr', _optimizer.param_groups[0]['lr'], step=_epoch)
        # log losses
        for _i, _n in enumerate(_loss_fn._names): _logger.log('train_' + _n, _loss_res[_i].item(), step=_epoch)
        _logger.log('train_all_loss', _loss_res[-1].item(), step=_epoch)
        # log metrics
        for _k, _v in _metrics_res.items(): _logger.log('train_' + _k, _v, step=_epoch)
        if _start_time is not None:
            _logger.log('time (min)', (time.time() - _start_time)/60.0, step=_epoch)


@torch.no_grad()
def evaluate(_model, _loader, _eval_metrics, _device, _epoch, _logger, _config, _train=True):

    _split = 'train' if _train else 'test'
    _model.eval()
    
    _loss_fn = build_loss(_config)
    _loss_res = torch.zeros(len(_loss_fn) + 1)
    
    for _bid, _data in enumerate(_loader):            
        
        _data = {k: d.to(_device) for k, d in _data.items()}
        
        with torch.amp.autocast('cuda'):
            _feat_obj, _pred_obj, _pred_subj = _model(**_data)
            
        
        _losses, _losses_sum = _loss_fn(pred_obj=_pred_obj, 
                                        pred_subj=_pred_subj,
                                        b_obj=_model.module.os_extractor.b_subj.weight,
                                        b_subj=_model.module.os_extractor.b_obj.weight,
                                        # b_obj=_model.os_extractor.b_subj.weight,
                                        # b_subj=_model.os_extractor.b_obj.weight, 
                                        fmri_obj = _pred_obj,
                                        **_data)
        _loss_res += torch.hstack([_losses, _losses_sum]).cpu()
        
        _eval_metrics.update(pred_obj=_pred_obj, pred_subj=_pred_subj, **_data)
    
        if _losses_sum.isnan() or _losses_sum.isinf():
            raise ValueError(f"Loss is {_losses_sum.item()}, stopping {_split}set evaluating at step {_bid} epoch {_epoch}")
    
    _loss_res /= len(_loader)
    _metrics_res = _eval_metrics.compute()
    _eval_metrics.reset()
    
    if _logger is not None:
        # log losses
        for _i, _n in enumerate(_loss_fn._names): _logger.log(_split + '_' + _n, _loss_res[_i].item(), step=_epoch)
        # log metrics
        _logger.log(_split + '_all_loss', _loss_res[-1].item(), step=_epoch)
        for _k, _v in _metrics_res.items(): _logger.log(_split + '_' + _k, _v, step=_epoch)
    
    return _metrics_res


def main(config):
    
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
    output_path = os.path.join('.', 'results',  '%s'%(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    
    config.output_path = output_path
    logger = wandb_logger(config) if config.local_rank == 0 else None
    # logger = None # use this line if you dont want to use wandb
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        save_config(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # create dataset and dataloader
    train_set = NSDDataset(root=config.data_path, train=True, image_feature=config.use_img_feat,
                             beta_type=config.beta_type, patch_size=config.patch_size, subj_ids=[1,2,3,4,5,6,7,8])
    test_set = NSDDataset(root=config.data_path, train=False, image_feature=config.use_img_feat,
                             beta_type=config.beta_type, patch_size=config.patch_size, subj_ids=[1,2,3,4,5,6,7,8])
   
    print(f'Dataset size: {len(train_set)}\nNumber of voxels: {train_set.n_voxels}')
    
    sampler = torch.utils.data.DistributedSampler(train_set, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

    train_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler, 
                shuffle=(sampler is None), pin_memory=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=config.num_workers)

    # create model
    config.n_voxels = train_set.n_voxels
    model = create_BrainEncoder(fmri_encoder=config.fmri_encoder, fmri_encoder_ckpt=config.fmri_encoder_ckpt, 
                                dim_fmri=config.dim_embed, dim_obj=config.dim_object, dim_subj=config.dim_subject, n_ca_heads=config.num_ca_heads,
                                n_obj=config.n_object, n_subj=config.n_subject, use_img_feature=config.use_img_feat, config=config)   
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=False)

    param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)
    
    loss_fn = build_loss(config)
    metric_fn = build_metrics(config, device)
    
    saver = None
    eval_metric = config.eval_metric
    best_metric = best_epoch = None
    decreasing = True if eval_metric == 'ham' else False
    if config.local_rank == 0:
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, amp_scaler=loss_scaler,
            checkpoint_dir=output_path, recovery_dir=output_path, decreasing=decreasing, max_history=config.ckpt_hist)

    start_time = time.time()
    print('Start Training the fmri bases ... ...')
    
    for ep in range(config.num_epoch):
        if torch.cuda.device_count() > 1: 
            sampler.set_epoch(ep) # to shuffle the data at every epoch
        train_one_epoch(model, train_loader, optimizer, device, ep, loss_fn, metric_fn, loss_scaler, logger, config, start_time)
        eval_res = evaluate(model, test_loader, metric_fn, device, ep, logger, config, _train=False) 
                
        if saver is not None:
            # save proper checkpoint with eval metric
            best_metric, best_epoch = saver.save_checkpoint(ep, metric=eval_res[eval_metric])
        
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    
    if logger is not None and best_metric is not None:
        logger.log('Best metric:', best_metric, step=config.num_epoch-1)
        logger.log('Best epoch:', best_epoch, step=config.num_epoch-1)
        logger.finish()
    

if __name__ == '__main__':

    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ['WANDB_DIR'] = "."

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    args = get_args_parser()
    args = args.parse_args()
    config = Config_BrainEncoder_Bases()
    config = update_config(args, config)
    main(config)
    