import torch
import torch.nn.functional as F
assert torch.__version__ == '1.2.0'
# other wise change the min function output

def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)

def min_loss(output, target, alpha=0, mode='per gene'):
    """
    selects one closest cell and computes the loss

    the target is the set of velocity target candidates, 
    find the closest in them.

    output: torch.tensor e.g. (128, 2000)
    target: torch.tensor e.g. (128, 30, 2000)
    """
    distance = torch.pow(target - torch.unsqueeze(output, 1), exponent=2) # (128, 30, 2000)
    if mode == 'per gene':
        distance = torch.min(distance, dim=1)[0]  # find the closest target for each gene (128, 2000)
        min_distance = torch.sum(distance, dim=1) # (128,)
    elif mode == 'per cell':
        distance =  torch.sum(distance, dim=2)# (128, 30)
        min_distance = torch.min(distance, dim=1)[0] # (128,)
    else:
        raise NotImplementedError

    # loss = torch.mean(torch.max(torch.tensor(alpha).float(), min_distance))
    loss = torch.mean(min_distance)
    
    return loss

def sim_loss():
    """the loss measuring similarities among input vectors"""
    return