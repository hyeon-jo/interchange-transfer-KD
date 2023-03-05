import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 

    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos

# TODO: Rename loss classes
# KLDivLoss => HintonLoss
# KDLoss => FitnetLoss
# RPNProjectionLoss => ProjectionLoss
class KLDivLoss(nn.Module):
    def __init__(self, code_weights, lambda_=0.5, T=1.0, box_kd=True):
        super(KLDivLoss, self).__init__()
        self.lambda_ = lambda_
        self.T = T
        self.box_kd = box_kd
        self.code_weights = code_weights

    def forward(self, teacher, student):
        kd_loss = float(0)
        for t, s in zip(teacher.items(), student.items()):
            kd_loss += nn.KLDivLoss(reduction='mean')(F.log_softmax(s[1]/self.T, dim=1),
                         F.softmax(t[1]/self.T, dim=1)) * (self.T * self.T)
        return kd_loss

class KDLoss(nn.Module):
    def __init__(self, p, in_channel, out_channel):
        super(KDLoss, self).__init__()
        self.p = p
        if not (in_channel is None and out_channel is None):
            self.channel_reduction = nn.Conv1d(in_channel, out_channel, kernel_size=1)
        else:
            self.channel_reduction = None

    def forward(self, f_t, f_s, norm=False):
        t_size = f_t.size()
        s_size = f_s.size()

        f_t = f_t.view(t_size[0], t_size[1], -1)
        f_s = f_s.view(s_size[0], s_size[1], -1)

        if not self.channel_reduction is None:
            f_t = self.channel_reduction(f_t)

        return (self.at(f_s, norm) - self.at(f_t, norm)).pow(2).mean()

    def at(self, f, norm):
        if norm:
            return F.normalize(f.pow(self.p)).mean(1).view(f.size(0), -1)

        return f.pow(self.p).mean(1).view(f.size(0), -1)


class HeadLoss(nn.Module):
    def __init__(self):
        super(HeadLoss, self).__init__()

    def forward(self, f_t, f_s):
        return F.normalize((f_s - f_t).abs()).mean()


class RPNProjectionLoss(nn.Module):
    def __init__(self, p, loss_type):
        super(RPNProjectionLoss, self).__init__()
        self.p = p
        self.loss_type = loss_type

    def forward(self, f_t, f_s, norm=True):
        s_size = f_s.size()
        t_size = f_t.size()

        # MaxPooling
        if self.loss_type == 'maxpool':
            f_s = F.adaptive_max_pool2d(f_s, (s_size[2], 1)) * F.adaptive_max_pool2d(f_s, (1, s_size[3]))
            f_t = F.adaptive_max_pool2d(f_t, (t_size[2], 1)) * F.adaptive_max_pool2d(f_t, (1, t_size[3]))
        elif self.loss_type == 'avgpool':
            f_s = F.adaptive_avg_pool2d(f_s, (s_size[2], 1)) * F.adaptive_avg_pool2d(f_s, (1, s_size[3]))
            f_t = F.adaptive_avg_pool2d(f_t, (t_size[2], 1)) * F.adaptive_avg_pool2d(f_t, (1, t_size[3]))

        return (self.at(f_s, norm) - self.at(f_t, norm)).pow(2).mean()

    def at(self, f, norm=True):
        if norm:
            return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

        return f.pow(self.p).mean(1).view(f.size(0), -1)