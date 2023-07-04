import torch
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['KL_divergence']

def KL_divergence(logits_p, logits_q):
    # p = softmax(logits_p)
    # q = softmax(logits_q)
    # KL(p||q)
    # suppose that p/q is in shape of [bs, num_classes]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p = F.softmax(logits_p, dim=1)
    q = F.softmax(logits_q, dim=1)

    shape = list(p.size())
    _shape = list(q.size())
    assert shape == _shape
    num_classes = shape[1]
    epsilon = 1e-8
    _p = (p + epsilon * Variable(torch.ones(*shape).to(device))) / (1.0 + num_classes * epsilon)
    _q = (q + epsilon * Variable(torch.ones(*shape).to(device))) / (1.0 + num_classes * epsilon)
    alpha = torch.mean(torch.sum(_p * torch.log(_p / _q), 1))
    # return torch.mean(torch.sum(_p * torch.log(_p / _q), 1))
    return alpha