from torch.nn.utils import clip_grad

from .hook import Hook


class OptimizerHook(Hook):
    def __init__(self, grad_clip=None, update_iter=4):
        self.grad_clip = grad_clip
        self.update_iter = update_iter
        self.iter = -1

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip
        )

    def after_train_iter(self, trainer):
        if self.iter == -1:
            trainer.optimizer.zero_grad()
            self.iter = 0
        self.iter += 1
        if self.iter == self.update_iter:
            trainer.optimizer.zero_grad()
        # print(trainer.outputs["loss"])
        trainer.outputs["loss"].backward()
        if self.iter == self.update_iter:
            if self.grad_clip is not None:
                self.clip_grads(trainer.model.parameters())
            trainer.optimizer.step()
            self.iter = 0
