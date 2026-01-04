from torch.utils.data import Dataset
import torch
import h5py
import numpy as np
from os import path
from math import ceil


def wrap_padding(x, length):
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[-1]), 'wrap').reshape(1, -1)


class NSDDataset(Dataset):
    def __init__(self, root='./nsd', beta_type='all', patch_size=16, train=True, image_feature=None, transform=None,subj_ids=[1,2,3,4,5,6,7,8], target_n_voxel=None):
        # fmri data
        fmri_root = path.join(root, 'fmri', beta_type)
        self.fmri_files = [h5py.File(path.join(fmri_root, f'subj0{i}_fp32_renorm.hdf5'), 'r') 
                           for i in subj_ids]
        split = 'train' if train else 'test'
        self.fmri_meta = torch.load(path.join(fmri_root, f'betas_{split}_meta.pt'))
        self.fmri_meta = [x for x in self.fmri_meta if x[0] in subj_ids]
        # image data
        image_root = path.join(root, 'image')
        self.image_label_file = torch.load(path.join(image_root, 'coco_73k_categories.pt'))
        self.use_img_feat = image_feature is not None
        if image_feature is not None:
            self.image_feature_file = h5py.File(path.join(image_root, 'coco_images_224_float16_clip_feat.hdf5'), 'r')
        
        self.subj_ids = subj_ids
        self.patch_size = patch_size
        self.n_voxels = patch_size * ceil(max([v['betas'].shape[-1] for v in self.fmri_files]) / patch_size) if target_n_voxel is None else target_n_voxel
        self.n_objs = self.image_label_file.shape[-1]
        self.n_subjs = len(self.fmri_files)

        self.transform = transform
        
        
    def __len__(self):
        
        return len(self.fmri_meta)

    def __getitem__(self, idx):
        
        gt_subj, fmri_idx, img_idx = self.fmri_meta[idx]
        voxel = self.fmri_files[self.subj_ids.index(gt_subj)]['betas'][fmri_idx]
        
        voxel_padded = wrap_padding(voxel, self.n_voxels)
        
        if self.use_img_feat:
            img_feat = self.image_feature_file['clip_feat'][img_idx]
        obj = self.image_label_file[img_idx].float()
            
        sample = {
            'gt_obj': obj,
            'gt_subj': torch.tensor(self.subj_ids.index(gt_subj), dtype=torch.int).type(torch.LongTensor),
            'voxels': voxel_padded,
        }
        if self.use_img_feat:
            sample['img_feats'] =  torch.tensor(img_feat[1:]) # only use local features
        return sample