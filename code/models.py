import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import torch.nn.functional as F


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
        # height (== width) for the new position embedding
        new_size = int(num_patches)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %d to %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size))
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed



class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, n_voxels=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = n_voxels // patch_size
        self.patch_shape = patch_size
        self.n_voxels = n_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels
        # assert V == self.n_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.n_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x


class fMRIEncoder(nn.Module):
    def __init__(self, n_voxels=224, patch_size=16, embed_dim=1024, in_chans=1,
                 depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, global_pool=False):
        super().__init__()
        self.patch_embed = PatchEmbed1D(n_voxels, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x += self.pos_embed
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        x = self.norm(x)

        return x  

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # B, n_seq, embed_dim
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # B, n_seq, embed_dim
    
    def load_checkpoint(self, ckpt):
        state_dict = torch.load(ckpt)['state_dict']
        
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        interpolate_pos_embed(self, state_dict)
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', [k for k in m if 'decoder' not in k])
        print('unexpected keys:', m)
        
     
class OSExtractor(nn.Module):
    
    def __init__(self, dim_fmri, dim_obj, dim_subj):
        super().__init__()
        
        assert dim_fmri == dim_obj + dim_subj
        
        self.b_obj = nn.Linear(dim_fmri, dim_obj, bias=False)
        self.b_subj = nn.Linear(dim_fmri, dim_subj, bias=False)
        
    def forward(self, X):
        
        X_obj = self.b_obj(X)
        X_subj = self.b_subj(X)
        
        return X_obj, X_subj
    
    
class OSExtractor_gate(nn.Module):
    
    def __init__(self, n_patches, obj_ratio_min=0.05):
        super().__init__()
        
        self.obj_ratio = nn.Parameter(torch.tensor(.5))
        self.idx = nn.Parameter(torch.arange(n_patches).float(), requires_grad=False)
        self.n_patches = n_patches
        self.obj_ratio_min = obj_ratio_min
        
    def forward(self, X):

        split_idx = torch.clamp(self.obj_ratio, self.obj_ratio_min, 1 - self.obj_ratio_min) * self.n_patches
        obj_mask = F.sigmoid(split_idx - self.idx).round().unsqueeze(1)
        X_obj =  obj_mask * X 
        X_subj = (1 - obj_mask) * X 
        
        return X_obj, X_subj 


class BrainEncoderBases(nn.Module):
    
    def __init__(self, fmri_encoder, dim_fmri,dim_obj, dim_subj, n_obj, n_subj, n_heads, g_pool=True, use_img_feature=False, norm_layer=nn.LayerNorm):
        super().__init__()
        
        assert dim_fmri == dim_obj + dim_subj
        
        self.fmri_encoder = fmri_encoder
        self.os_extractor = OSExtractor(dim_fmri, dim_obj, dim_subj)
        self.obj_norm1 = norm_layer(dim_obj)
        self.g_pool = g_pool
        
        self.use_img_feature = use_img_feature
        if use_img_feature:
            self.cross_attn = nn.MultiheadAttention(dim_fmri, num_heads=n_heads, dropout=0.2, bias=True, add_bias_kv=False, kdim=dim_obj, vdim=dim_obj, batch_first=True)
            self.obj_norm2 = norm_layer(dim_fmri)
            self.fc_obj = nn.Linear(dim_fmri, n_obj)
        else:
            self.fc_obj = nn.Linear(dim_obj, n_obj)

        self.fc_subj = nn.Linear(dim_subj, n_subj)

    def forward(self, voxels, img_feats=None, need_weights=False, *args, **kwargs):
        
        x = self.fmri_encoder(voxels)
        x_obj, x_subj = self.os_extractor(x)
        x_obj = self.obj_norm1(x_obj)
        if self.use_img_feature:
            x_obj, attn = self.cross_attn(query=img_feats, key=x_obj, value=x_obj, need_weights=need_weights)
            x_obj = self.obj_norm2(x_obj)
        
        if self.g_pool:
            x_obj = x_obj.mean(1)
            x_subj = x_subj.mean(1)
        
        x_obj_cls = self.fc_obj(x_obj)
        x_subj_cls = self.fc_subj(x_subj)
        
        return x_obj, x_obj_cls, x_subj_cls