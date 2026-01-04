from torch.nn.modules.loss import _Loss
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
from torchmetrics.functional import pairwise_cosine_similarity

class ObjectLoss(BCEWithLogitsLoss):
       
    def forward(self, pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
        return super().forward(pred_obj, gt_obj)
    
    
class SubjectLoss(CrossEntropyLoss):
    
    def forward(self, pred_subj: torch.Tensor, gt_subj: torch.Tensor) -> torch.Tensor:
        return super().forward(pred_subj, gt_subj)
    

class AlignmentLoss(MSELoss):
    
    def forward(self, fmri_obj: torch.Tensor, img_obj: torch.Tensor) -> torch.Tensor:
        
        fmri_sim = pairwise_cosine_similarity(fmri_obj)
        img_sim = pairwise_cosine_similarity(img_obj)
        
        return super().forward(fmri_sim, img_sim)


class BasesLoss(_Loss):
    
    def forward(self, b_subj: torch.Tensor, b_obj: torch.Tensor) -> torch.Tensor:
        
        b = torch.vstack([b_subj, b_obj])
        hidden_d = b.size(1)
        bbt = b @ b.T
        
        return torch.norm(bbt - torch.eye(hidden_d, device=b.device), p='fro')
    

class StackLoss(nn.Module):
    def __init__(self, **kwargs):
        '''
        kwarg[i] = name : (criterion, weight, [params])
        '''
        super(StackLoss, self).__init__()
        self._args = kwargs
        self._names = []
        self._criterions = nn.ModuleList([])
        self._weights = []
        self._params = []
        
        for _n, _v in kwargs.items():
            self._names.append(_n)
            self._criterions.append(_v[0])
            self._weights.append(_v[1])
            self._params.append(_v[2])
        
        self._weights = torch.tensor(self._weights)        
        self._loss_units = None

    @property
    def loss_units(self): return self._loss_units
        
    def __len__(self): return len(self._names)
    
    
    def forward(self, **kwargs):
        _criterion_outputs = [_criterion(**{k:v for k, v in kwargs.items() if k in _param})
                              for _criterion, _param in zip(self._criterions, self._params)]
        self._loss_units = _criterion_outputs
        _weighted_criterion_outputs = torch.stack([_w * _o for _w, _o in zip(self._weights, _criterion_outputs)])
        
        return _weighted_criterion_outputs, _weighted_criterion_outputs.sum()
        
LOSS_dict = dict(
    obj_loss=(ObjectLoss,       ('pred_obj', 'gt_obj')),
    subj_loss=(SubjectLoss,     ('pred_subj', 'gt_subj')),
    bases_loss=(BasesLoss,      ('b_subj', 'b_obj')),
    align_loss=(AlignmentLoss,  ('fmri_obj', 'img_obj')),
) 
 
if __name__ == '__main__':
    n_classes = 10
    n_subj = 5
    b = 50
    loss = StackLoss(
        obj_loss=(ObjectLoss(), 0.1, ('pred_obj', 'gt_obj')),
        subj_loss=(SubjectLoss(), 0.1, ('pred_subj', 'gt_subj')),
        bases_loss=(BasesLoss(), .1, ('subj_b', 'obj_b')),
        )
    
    
    
    
    pred_subj = torch.randn(b, n_subj)
    gt_subj = torch.randint(0, n_subj, size=(b, ))

    pred_obj = torch.randn(b, n_classes)
    gt_obj = torch.ones(b, n_classes, dtype=torch.float32)
    b_subj = torch.randn(b, 10, 3)
    b_obj = torch.randn(b, 10, 3)

    

    print(loss(
        pred_subj=pred_subj, gt_subj=gt_subj,
        pred_obj=pred_obj, gt_obj=gt_obj, 
               subj_b=b_subj, obj_b=b_obj
        ),
          )
    
    
           
