from torch import nn
from torch import optim

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=1e-1)
steps = 35
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0)

for epoch in range(1):
    for idx in range(steps):
        print('Pre ', epoch, idx, f'{scheduler.get_last_lr()[0]:e}')
        scheduler.step()
        # print('Post', epoch, idx, scheduler.get_last_lr())

    print('Reset scheduler')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-7, last_epoch=-1)