# optimizer.py
import torch

class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, alpha=0.5, k=5):
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self._counter = 0
        self.slow_weights = [p.clone().detach().to(p.device) for group in self.param_groups for p in group['params'] if p.requires_grad]
        for w in self.slow_weights:
            w.requires_grad = False

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._counter += 1
        if self._counter % self.k != 0:
            return loss
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    slow = self.slow_weights[idx]
                    slow.data += self.alpha * (p.data - slow.data)
                    p.data = slow.data.clone()
                    idx += 1
        return loss

def get_optimizer_and_scheduler(model, config, steps_per_epoch):
    opt_name = config["optimizer"]
    sch_name = config["scheduler"]

    if "adamw" in opt_name:
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    elif "adam" in opt_name:
        base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif "sgd" in opt_name:
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    if "lookahead" in opt_name:
        optimizer = Lookahead(base_optimizer)
    else:
        optimizer = base_optimizer

    if sch_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            base_optimizer, max_lr=1e-3, epochs=100, steps_per_epoch=steps_per_epoch)
    elif sch_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {sch_name}")

    return optimizer, scheduler
