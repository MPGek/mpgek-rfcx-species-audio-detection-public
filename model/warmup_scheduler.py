import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)



if __name__ == '__main__':
    # 'params': {'lr': 2e-3, 'total_steps': 18 * 64, 'warmup_proportion': 0.3, 'min_lr': 1e-6},
    v = torch.zeros(10)
    optim = torch.optim.Adam([v], lr=2e-3)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100-6, eta_min=1e-7, last_epoch=-1)
    scheduler = cosine_scheduler
    scheduler = GradualWarmupScheduler(optim, multiplier=2, total_epoch=10, after_scheduler=cosine_scheduler)
    a = []
    b = []

    a.append(optim.param_groups[0]['lr'])
    b.append(0)
    print(0, optim.param_groups[0]['lr'])

    for iteration in range(1, 101):
        scheduler.step(iteration)
        a.append(iteration)
        b.append(optim.param_groups[0]['lr'])
        print(iteration, optim.param_groups[0]['lr'])

    plt.plot(a, b)
    plt.show()
