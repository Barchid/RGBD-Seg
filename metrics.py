import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class M_IOU(object):
    def __init__(self, n_classes, ignore_index=255):
        self.n_classes = n_classes
        self.hist = torch.zeros(n_classes, n_classes).to(device).detach()
        self.ignore_index = ignore_index

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = logits.clone().detach()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # avoid void
        keep = targets != self.ignore_index

        self.hist += torch.bincount(targets[keep] * self.n_classes + preds[keep], minlength=self.n_classes ** 2).view(self.n_classes, self.n_classes).float()

    def compute(self):
        ious = self.hist.diag() / (self.hist.sum(dim=0) + self.hist.sum(dim=1) - self.hist.diag())
        miou = ious.mean()
        return miou.item()
