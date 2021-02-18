import torch
from torch.optim.lr_scheduler import _LRScheduler

from pytorch_toolbelt.optimization.lr_schedules import CosineAnnealingLRWithDecay


class RectifiedWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_proportion, total_epoch, min_lr, log_scale=False, log_weight=0):
        self.warmup_proportion = warmup_proportion
        self.total_epoch = total_epoch
        self.min_lr = min_lr
        self.log_scale = log_scale
        self.log_lr_multiplier = None
        self.log_weight = log_weight

        self.asc_epochs = int(total_epoch * warmup_proportion)
        self.desc_epochs = total_epoch - self.asc_epochs - 1

        super().__init__(optimizer)

        # optim.param_groups[0]['lr']

    def get_lr(self):
        if self.log_lr_multiplier is None:
            self.log_lr_multipliers = [(self.min_lr / base_lr) ** (1.0 / self.desc_epochs) for base_lr in self.base_lrs]

        if self.last_epoch <= self.asc_epochs:
            lrs = [base_lr * self.last_epoch / self.asc_epochs for base_lr in self.base_lrs]
        else:
            desc_step = self.last_epoch - self.asc_epochs

            linear_scale = desc_step / self.desc_epochs
            lrs = [base_lr - (base_lr - self.min_lr) * linear_scale for base_lr in self.base_lrs]

            if self.log_scale:
                log_lrs = [base_lr * (lr_mul ** desc_step) for base_lr, lr_mul in
                           zip(self.base_lrs, self.log_lr_multipliers)]

                if self.log_weight == 1:
                    lrs = [(lr1 + lr2) / 2 for lr1, lr2 in zip(lrs, log_lrs)]
                elif self.log_weight == 2:
                    lrs = [lr1 * (1 - linear_scale) + lr2 * linear_scale for lr1, lr2 in zip(lrs, log_lrs)]
                else:
                    lrs = log_lrs

            lrs = [max(self.min_lr, lr) for lr in lrs]

        return lrs

    def step(self, metrics=None, epoch=None):
        super().step(epoch)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 'params': {'lr': 2e-3, 'total_steps': 18 * 64, 'warmup_proportion': 0.3, 'min_lr': 1e-6},
    # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100-6, eta_min=1e-7, last_epoch=-1)
    # scheduler = cosine_scheduler
    # scheduler = GradualWarmupScheduler(optim, multiplier=2, total_epoch=10, after_scheduler=cosine_scheduler)

    optim = [
        torch.optim.Adam([torch.zeros(10)], lr=1e-3),
        torch.optim.Adam([torch.zeros(10)], lr=1e-3),
        torch.optim.Adam([torch.zeros(10)], lr=1e-3),
        torch.optim.Adam([torch.zeros(10)], lr=1e-3),
        torch.optim.Adam([torch.zeros(10)], lr=1e-3),
    ]
    total_epoch = 141
    scheduler = {
        # 'rect': RectifiedWarmupScheduler(optim[0], 0.1, total_epoch, min_lr=1e-7, log_scale=False),
        # 'log': RectifiedWarmupScheduler(optim[1], 0.1, total_epoch, min_lr=1e-7, log_scale=True),
        'cWarm': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim[0], 20, 2),
        'cDecay': CosineAnnealingLRWithDecay(optim[0], 20, 1),
        'log_w1': RectifiedWarmupScheduler(optim[2], 0.1, total_epoch, min_lr=1e-7, log_scale=True, log_weight=1),
        'log_w2': RectifiedWarmupScheduler(optim[3], 0.1, total_epoch, min_lr=1e-7, log_scale=True, log_weight=2),
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR(optim[4], total_epoch)
    }
    a = []
    b = [[] for _ in scheduler]

    a.append(0)
    for i in range(len(scheduler)):
        b[i].append(optim[i].param_groups[0]['lr'])
    print(0, *[o.param_groups[0]['lr'] for o in optim])

    for iteration in range(1, total_epoch):
        for sch in scheduler.values():
            sch.step()

        a.append(iteration)
        for i in range(len(scheduler)):
            b[i].append(optim[i].param_groups[0]['lr'])
        print(iteration, *[o.param_groups[0]['lr'] for o in optim])

    for idx, name in enumerate(scheduler.keys()):
        plt.plot(a, b[idx], label=name)
    plt.legend()
    # plt.yscale('log')
    plt.show()
