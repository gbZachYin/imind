import yaml
from os.path import join, split
from models import fMRIEncoder, BrainEncoderBases
import math
from loss import LOSS_dict, StackLoss
from metrics import *
from collections import OrderedDict

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


def save_config(config, path):
    print(config.__dict__)
    with open(join(path, 'config.yml'), 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
        

def create_BrainEncoder(fmri_encoder='mae', fmri_encoder_ckpt=None, n_voxels=17920,
                        dim_fmri=1024, dim_obj=1000, dim_subj=24, 
                        n_obj=80, n_subj=8, n_ca_heads=1, use_img_feature=False, config=None):
    if fmri_encoder == 'mae':
        print(fmri_encoder, fmri_encoder_ckpt)
        fmri_features = create_fMRIEncoder(ckpt=fmri_encoder_ckpt, n_voxels=n_voxels, config=config)
    else:

        raise ValueError(f'fmri_encoder type {fmri_encoder} not specified')
    

    return BrainEncoderBases(
        fmri_encoder=fmri_features, 
        dim_fmri=dim_fmri, dim_obj=dim_obj, dim_subj=dim_subj,
        n_obj=n_obj, n_subj=n_subj, n_heads=n_ca_heads,
        g_pool=True, use_img_feature=use_img_feature
        )
    

def create_fMRIEncoder(ckpt=None, n_voxels=17920, **kwargs):
    
    if ckpt is not None:

        with open(join(split(ckpt)[0], 'config.yml'), 'r') as f:
            mae_config = yaml.safe_load(f)
        m = fMRIEncoder(n_voxels=n_voxels,
                        patch_size=mae_config['patch_size'], 
                        embed_dim=mae_config['embed_dim'], 
                        depth=mae_config['depth'],
                        num_heads=mae_config['num_heads'], 
                        mlp_ratio=mae_config['mlp_ratio'])
        m.load_checkpoint(ckpt)
    else:
        mae_config = kwargs['config']
        m = fMRIEncoder(n_voxels=n_voxels, 
                        patch_size=mae_config.patch_size, 
                        embed_dim=mae_config.embed_dim, 
                        depth=mae_config.depth,
                        num_heads=mae_config.num_heads, 
                        mlp_ratio=mae_config.mlp_ratio)
    return m

def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs 
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def build_loss(config, reduction='mean'):
    
    _loss_dict = OrderedDict([(k, (LOSS_dict[k][0](reduction=reduction), 
                                   getattr(config, f'{k}_w'),
                                   LOSS_dict[k][1])) for k in config.losses])
    return StackLoss(**_loss_dict)


def build_metrics(config, device):
    
    return EvalMetrics(
        auc=(ObjectAUROC(config.n_object).to(device),            ('pred_obj' , 'gt_obj')),
        ham=(ObjectHamming(config.n_object).to(device),          ('pred_obj' , 'gt_obj')),
        map=(ObjectAveragePrecision(config.n_object).to(device), ('pred_obj' , 'gt_obj')),
        acc=(SubjectAccuracy(config.n_subject).to(device),       ('pred_subj', 'gt_subj')),
        mcc=(SubjectMCC(config.n_subject).to(device),            ('pred_subj', 'gt_subj')),
    )