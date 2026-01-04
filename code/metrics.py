from collections import OrderedDict
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAUROC, MultilabelHammingDistance
from torchmetrics.classification import MulticlassMatthewsCorrCoef, MulticlassAccuracy
from torch.nn.functional import sigmoid
import torch
    
    
class ObjectAveragePrecision(MultilabelAveragePrecision):
    
    def update(self, pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
        return super().update(sigmoid(pred_obj), gt_obj.int())
    
    
class ObjectHamming(MultilabelHammingDistance):
    
    def update(self, pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
        return super().update(sigmoid(pred_obj), gt_obj)


class ObjectAUROC(MultilabelAUROC):
    
    def update(self, pred_obj: torch.Tensor, gt_obj: torch.Tensor) -> torch.Tensor:
        return super().update(sigmoid(pred_obj), gt_obj.int())


class SubjectAccuracy(MulticlassAccuracy):
    
    def update(self, pred_subj: torch.Tensor, gt_subj: torch.Tensor) -> torch.Tensor:
        return super().update(pred_subj, gt_subj)


class SubjectMCC(MulticlassMatthewsCorrCoef):
    
    def update(self, pred_subj: torch.Tensor, gt_subj: torch.Tensor) -> torch.Tensor:
        return super().update(pred_subj, gt_subj)


class EvalMetrics:
    
    def __init__(self, **kwargs):
        '''
        arg[i] = (name, metric, [params])
        '''
        super().__init__()
        self._args = kwargs
        
        
    def update(self, **kwargs):
        for _name, (_metric, _param) in self._args.items(): _metric.update(**{k:v for k, v in kwargs.items() if k in _param})

    
    def compute(self):
        return OrderedDict([(_name, _metric.compute().item()) for _name, (_metric, _) in self._args.items()])
    
    def reset(self):
        for _metric, _ in self._args.values(): _metric.reset()
    

if __name__ == '__main__':
    n_classes = 10
    n_subj = 5
    
    em = EvalMetrics(
        auc=(ObjectAUROC(n_classes), ('pred_obj', 'gt_obj')),
        ham=(ObjectHamming(n_classes), ('pred_obj', 'gt_obj')),
        map=(ObjectAveragePrecision(n_classes), ('pred_obj', 'gt_obj')),
        acc=(SubjectAccuracy(n_subj), ('pred_subj', 'gt_subj')),
        mcc=(SubjectAccuracy(n_subj), ('pred_subj', 'gt_subj')),
    )
    
    for b in [50, 50, 5]:
        pred_subj = torch.randn(b, n_subj)
        gt_subj = torch.randint(0, n_subj, size=(b, ))

        pred_obj = torch.randn(b, n_classes)
        gt_obj = torch.ones(b, n_classes, dtype=torch.int)
        # gt_obj[0] = 0

        em.update(pred_obj=pred_obj, gt_obj=gt_obj, pred_subj=pred_subj, gt_subj=gt_subj)
    
    print(em['auc'])
    # em.reset()
    # print(em.compute())
    
    
    