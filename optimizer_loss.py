# FROM https://github.com/MichaelFan01/STDC-Seg
import torch


class Optimizer(object):
    def __init__(self,
                 model,
                 loss,
                 lr0,
                 momentum,
                 wd,
                 warmup_steps,
                 warmup_start_lr,
                 max_iter,
                 power,
                 *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        loss_nowd_params = loss.get_params()
        # print(wd_params)
        # print(nowd_params)
        # print(loss_nowd_params)
        # exit(0)
        param_list = [
                {'params': wd_params},
                {'params': nowd_params, 'weight_decay': 0},
                {'params': lr_mul_wd_params, 'lr_mul': True},
                {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr_mul': True},
                {'params': loss_nowd_params}]
        # {'params': loss_nowd_params, 'weight_decay': 0, 'lr': 0.000001}]
        self.optim = torch.optim.SGD(
                param_list,
                lr=lr0,
                momentum=momentum,
                weight_decay=wd)
        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)

    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr

    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            print('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()
