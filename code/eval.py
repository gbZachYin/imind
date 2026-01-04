import os
import torch
from torch.utils.data import DataLoader
from utils import create_BrainEncoder, build_metrics
from metrics import *
from dataset import NSDDataset
from os.path import join, split
import yaml
from models import fMRIEncoder, BrainEncoderBases

def create_model_and_loader(_ckpt, is_trainset=False, root='..', batch_size=500, subj_ids=[1,2,3,4,5,6,7,8], device='cuda:0'):
    sd = torch.load(_ckpt)['state_dict']
    with open(join(split(_ckpt)[0], 'config.yml'), 'r') as f:
        config = yaml.safe_load(f)
    
        
    f_encoder = fMRIEncoder(n_voxels=17920,
                            patch_size=config['patch_size'], 
                            embed_dim=config['embed_dim'], 
                            depth=config['depth'],
                            num_heads=config['num_heads'], 
                            mlp_ratio=config['mlp_ratio'])
    
    model = BrainEncoderBases(fmri_encoder=f_encoder, 
                            dim_fmri=config['dim_embed'],  
                            dim_obj=config['dim_object'],
                            dim_subj=config['dim_subject'],
                            n_obj=config['n_object'], 
                            n_subj=config['n_subject'], 
                            n_heads=config['num_ca_heads'],
                            g_pool=True,
                            use_img_feature=config['use_img_feat'])
    model.load_state_dict(sd)
    model = model.eval().cpu().to(device)
    
    dataset = NSDDataset(root=root, train=is_trainset, image_feature=config['use_img_feat'],
                             beta_type=config['beta_type'], patch_size=config['patch_size'],subj_ids=subj_ids, target_n_voxel=17920)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=config['num_workers'])
    
    
    return model, loader, config


@torch.no_grad()
def model_forward(_model, _loader, _config, _device='cuda:0'):

    _model.eval()   
    _eval_metrics = EvalMetrics(
        auc=(ObjectAUROC(_config['n_object']).cuda(),            ('pred_obj' ,  'gt_obj')),
        ham=(ObjectHamming(_config['n_object']).cuda(),          ('pred_obj' ,  'gt_obj')),
        map=(ObjectAveragePrecision(_config['n_object']).cuda(), ('pred_obj' ,  'gt_obj')),
        acc=(SubjectAccuracy(_config['n_subject']).cuda(),       ('pred_subj',  'gt_subj')),
        mcc=(SubjectMCC(_config['n_subject']).cuda(),       ('pred_subj',  'gt_subj')),
    )
    
    for _data in _loader:
        
        _data = {k: d.cuda() for k, d in _data.items()}
        
        with torch.amp.autocast('cuda'):
            _feat_obj, _pred_obj, _pred_subj = _model(**_data)
        
        _eval_metrics.update(pred_obj=_pred_obj, pred_subj=_pred_subj, **_data)
        
    _metrics_res = _eval_metrics.compute()
    _eval_metrics.reset()
    
    for _k, _v in _metrics_res.items(): print('test_' + _k, _v)
    

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    batch_size = 4000
    data_path = './nsd'
    ckpt = './saved_ckpt/imind/last.pth.tar'

    model, loader, config = create_model_and_loader(ckpt, False, data_path, batch_size)
    model_forward(model, loader, config)

