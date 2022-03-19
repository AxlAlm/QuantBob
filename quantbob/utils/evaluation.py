
"""
https://discuss.pytorch.org/t/spearmans-correlation/91931/6
https://forum.numer.ai/t/differentiable-spearman-in-pytorch-optimize-for-corr-directly/2287/22
https://forum.numer.ai/t/model-evaluation-metrics/337/6

https://pypi.org/project/torchsort/
"""
import torchsort

def spearmans(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()