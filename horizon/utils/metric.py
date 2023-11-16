import torch
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.regression.mae import MeanAbsoluteError


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mean_ap(output, target, labels=26):
    metric = MultilabelAveragePrecision(num_labels=labels,
                                        average="macro",
                                        thresholds=None)
    return metric(output, target)


def mean_aae(output, target):
    metric = MeanAbsoluteError().to(output.device)
    return metric(output, target)
