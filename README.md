# iMIND: Insightful Multi-subject Invariant Neural Decoding (NeurIPS 2025)

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2509.17313) 
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://zachyin.com/imind/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-yellow)](https://huggingface.co/datasets/zachyin/imind/tree/main)

This is the official implementation of **iMIND**, presented at **NeurIPS 2025**. iMIND introduces a dual-decoding framework for novel multi-subject invariant neural decoding. By combining the representative power of Masked Autoencoders with a specialized orthonormal disentanglement strategy, iMIND effectively bridges the gap between individual neural variability and universal semantic decoding.

![imind](poster.png)

## 📢 News
* **Jan. 2026:** 🚀 Code and checkpoints are released!
* **Dec. 2025:** Poster and project website are live.
* **Sep. 2025:** 🎉 **iMIND** was accepted at NeurIPS 2025!
* **Nov. 2024:** Project initiated.

---

## 🛠️ Environment Setup

We recommend using **Conda** for environment management to ensure dependency stability.

```bash
# Create and activate environment
conda create --name imind python=3.8.18
conda activate imind

# Install PyTorch (adjust cuda version as needed for your hardware)
pip install torch torchvision

# Install remaining dependencies
pip install -r requirements.txt
```

## 💾 Data & Checkpoints

### 1. Dataset Access
The **Natural Scenes Dataset (NSD)** is the primary dataset used in this work. To access the data, you must follow the official NSD protocol:
* **Terms of Use:** Review and agree to the [NSD Terms and Conditions](https://cvnlab.slite.com/api/s/note/9dgh5HCqgZYhMoAESZBS86/Terms-and-Conditions).
* **Access Form:** Complete the [NSD Data Access Form](https://forms.gle/eT4jHxaWwYUDEf2i9) to request the raw or pre-processed data.

### 2. iMIND Assets & Weights
We provide pre-processed fMRI data (compatible with our pipeline), CLIP visual features, and model checkpoints on Hugging Face.

| Asset | Description | Link |
| :--- | :--- | :--- |
| **Pre-processed Data** | Subject-wise 1D fMRI voxels and image features | [ Download](https://huggingface.co/datasets/zachyin/imind/tree/main) |
| **MAE Checkpoint** | Stage 1 pre-trained weights (Masked Autoencoder) | [Download](https://huggingface.co/datasets/zachyin/imind/tree/main/saved_ckpt/mae) |
| **Full iMIND Weights** | Stage 2 model for evaluation | [Download](https://huggingface.co/datasets/zachyin/imind/tree/main/saved_ckpt/imind_final) |

### 3. Recommended Directory Structure
To ensure the scripts run correctly without manual path configuration, please organize the downloaded assets as follows:
```
📂 code  
📂 nsd  
┣ 📂 fmri  
┃   ┣ 📂 avg  
┃   ┃   ┗ 📜 betas_test_meta.pt       # a list of (subj_id, fmri_id, img_id)
┃   ┃   ┗ 📜 betas_train_meta.pt      
┃   ┃   ┗ 📜 subj01_fp32_renorm.hdf5  # 1d fmri data for subj01
┃   ┃   ┗ 📜 subj02_fp32_renorm.hdf5  
┃   ┃   ┗ ...  
┃   ┃   ┗ 📜 subj07_fp32_renorm.hdf5  
┃   ┃   ┗ 📜 subj08_fp32_renorm.hdf5  
┣ 📂 image  
┃   ┗ 📜 coco_73k_categories_name.csv            # image object labels
┃   ┗ 📜 coco_73k_categories_name.txt            # image object names
┃   ┗ 📜 coco_73k_categories.pt                  # image object labels .pt
┃   ┗ 📜 coco_images_224_float16_clip_feat.hdf5  # image CLIP faetures
┃   ┗ 📜 coco_images_224_float16.hdf5            # nsd image
📂 results  
📂 saved_ckpt  
┣ 📂 imind  
┃   ┗ 📜 config.yml                              # config file
┃   ┗ 📜 last.pth.tar                            # imind visual+neural ckpt
┣ 📂 mae  
┃   ┗ 📜 config.yml                              # config file
┃   ┗ 📜 model.pth.tar                           # MAE pre-trained on NSD
```

## 🔥 Model Training

The iMIND framework is implemented as a two-stage training pipeline to achieve subject-invariant neural decoding.



### Stage 1: Masked Autoencoder (MAE) Pre-training
In the first stage, we pre-train a Vision Transformer-based MAE on fMRI data from all 8 subjects in the NSD dataset to build a neural data encoder.

* **Pre-trained Weights:** We have released the stage 1 checkpoint [here](https://huggingface.co/datasets/zachyin/imind/tree/main/saved_ckpt/mae).
* **Custom Training:** If you wish to perform pre-training yourself, please follow the protocol defined in [Mind-Vis](https://github.com/zjc062/mind-vis).

### Stage 2: Semantic & Biometric Dual Decoding
In the second stage, the model refines the learned neural representations and disentangles them into distinct biometric and semantic subspaces via orthonormal projection. This decomposition ensures that semantic features are isolated for object classification, while biometric features are leveraged for subject identification.

#### 1. Configuration
Before launching the training, update the paths and hyperparameters in `code/config.py`:

```python
class Config_BrainEncoder_Bases():
    def __init__(self):
        ...
        self.data_path = 'YOUR_DATA_ROOT'   # Path to pre-processed NSD data
        ...
        self.fmri_encoder_ckpt = './saved_ckpt/mae/model.pth.tar' # MAE 
```

#### 2. Single GPU Training:
Simply run below to start stage-2 training:
```bash
python code/main.py
```

Optionally, some hyperparameters can be specified via comand-line, but these arguments will override what you have in `config.py`:
```bash
python code/main.py --num_epoch 150 --batch_size 256 
```

#### 3. Multi-GPU Training (DDP)
```bash
export NCCL_P2P_DISABLE=1
python -m torch.distributed.launch --nproc_per_node=2 code/main.py
```

### 📈 Monitoring & Checkpoints
* **Checkpoints:** Models are saved locally to the `./results` directory.
* **Tracking:** Real-time training statistics are logged to your **WandB** dashboard.

## 🎯 Model Evaluation
To run inference or evaluate a checkpoint only, you should specify the path to it in `code/eval.py` and run the following:
```bash
python code/eval.py
```

## 💬 FAQs
To be updated ...
## ✍️ Citation
If you find this work or code useful for your research, please cite our paper. Thanks!
```
@inproceedings{yinmind,
  title={$ i $ MIND: Insightful Multi-subject Invariant Neural Decoding},
  author={Yin, Zixiang and Li, Jiarui and Ding, Zhengming},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
