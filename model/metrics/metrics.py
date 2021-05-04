import torch
from torch import nn

class metrics_func(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):

        y_pred=logits.view(size=(-1,)).float()
        y_true=labels.view(size=(-1,)).float()

        ones=torch.ones_like(y_true)
        zero=torch.zeros_like(y_true)
        y_true_one=torch.where(y_true<1,zero,ones)
        y_true_one=y_true_one.view(size=(-1,)).float()

        ones=torch.ones_like(y_pred)
        zero=torch.zeros_like(y_pred)
        y_pred_one=torch.where(y_pred<1,zero,ones)
        y_pred_one=y_pred_one.view(size=(-1,)).float()

        
        corr=torch.eq(y_pred,y_true)
        corr=torch.multiply(corr.float(),y_true_one)
        recall=torch.sum(corr)/(torch.sum(y_true_one)+1e-8)
        precision=torch.sum(corr)/(torch.sum(y_pred_one)+1e-8)
        f1=2*recall*precision/(recall+precision+1e-8)

        return recall, precision, f1
