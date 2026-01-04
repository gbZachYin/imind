class Config_BrainEncoder_Bases():
    def __init__(self):
        # Project setting
        self.project = 'imind'
        self.data_path = './nsd'
        self.num_workers = 32
        self.beta_type = 'avg' # avg means fmri data we used was averaged across sessions whose stimuli were the same
        self.seed = 2024
        self.use_img_feat = True
        self.fmri_encoder = 'mae'
        # self.fmri_encoder_ckpt = None
        self.fmri_encoder_ckpt = './saved_ckpt/mae/model.pth.tar'
        
        # Model Parameters
        self.patch_size = 64
        self.dim_embed = 768 # has to be a multiple of num_heads
        self.dim_subject = 68 
        self.dim_object = 700 
        self.embed_dim = self.dim_embed
        self.depth = 12 # encoder layers
        self.num_heads = 6 # encoder SA heads
        self.mlp_ratio = 1.0
        self.num_ca_heads = 4 # image-fmri cross attention heads
        self.n_object = 80 # number of objects for image modality
        self.n_subject = 8 # number of subjects for fmri modality
        # Training Parameters
        self.lr = 7.5e-4
        self.min_lr = 1e-6
        self.weight_decay = 0.05
        self.num_epoch = 100
        self.warmup_epochs = 10
        self.batch_size = 200
        self.clip_grad = 0.8
        self.accum_iter = 1
        self.ckpt_hist = 10
        self.eval_metric = 'map' # metric to define best model to save, can be changed other metrics
        self.losses = ['obj_loss', 'subj_loss', 'bases_loss']

        self.subj_loss_w = 1
        self.obj_loss_w = 1
        self.bases_loss_w = 0.1
        
        # distributed training
        self.local_rank = 0