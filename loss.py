# loss.py
import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        log_probs = self.log_softmax(x)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

def get_loss_function(name, num_classes):
    if name == "label_smoothing":
        return LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
    elif name == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")
