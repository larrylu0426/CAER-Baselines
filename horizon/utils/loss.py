import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


def weighted_euclidean_loss(output, target):
    mode = "batch"
    device = output.device
    if mode == "mean":
        weights = (torch.ones((1, 26)) / 26).to(device)
    elif mode == "global":
        # the weights is calculated on training set with "w = 1/(ln(c+p))" and c=1.2
        weights = torch.FloatTensor([
            4.5519, 5.2693, 5.1173, 2.7996, 5.3101, 3.1440, 5.1574, 4.2852,
            5.0167, 4.8626, 5.3262, 1.7856, 4.7093, 3.0443, 4.9650, 5.3011,
            2.7321, 5.2901, 4.1472, 3.9224, 5.0842, 5.1247, 5.2194, 5.0724,
            4.8143, 4.8552
        ]).unsqueeze(0).to(device)
    else:  # mode == "batch" w = 1/(ln(c+p))
        c = 1.2  # used in "Context-Aware Emotion Recognition Based on Visual Relationship Detection"
        p = target.float().mean(dim=0)
        weights = 1.0 / torch.log(p + c)
    loss = (((output - target)**2) * weights).mean(dim=1).mean()
    return loss


def margin_L2(output, target):
    theta = 1
    l1_distance = torch.abs(output - target)
    loss = F.mse_loss(output, target, reduce=False)
    loss[(l1_distance < theta)] = 0.0
    loss = loss.sum(dim=1).mean()
    return loss


def smooth_L1(output, target):
    # support for "v_k=1" in paper, natively
    return F.smooth_l1_loss(output, target, reduction='mean', beta=1) * 3