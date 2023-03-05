import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat
import matplotlib.pyplot as plt
import math

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
    loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)
    return loss


class SSLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(SSLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        batch_size = target.size(0)
        features = torch.cat([target, output], dim=0)
        ind = torch.cat([ind, ind], dim=0)
        preds = _transpose_and_gather_feat(features, ind)  # 2B x M x C
        p_size = preds.size()
        target = preds[batch_size:].view(batch_size, p_size[1], p_size[2])
        pred = preds[:batch_size].view(batch_size, p_size[1], p_size[2])
        mask = mask.float().unsqueeze(2)

        loss = F.mse_loss(pred * mask, target * mask, reduction='mean')
        return loss


class KDRegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(KDRegLoss, self).__init__()

    def forward(self, student, mask, ind, teacher):
        batch_size = teacher.size(0)
        features = torch.cat([teacher, student], dim=0)
        ind = torch.cat([ind, ind], dim=0)
        preds = _transpose_and_gather_feat(features, ind)
        target = preds[:batch_size]
        pred = preds[batch_size:]
        mask = mask.float().unsqueeze(2)

        loss = F.l1_loss(pred * mask, target * mask, reduction='none')
        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2, 0).sum()
        return loss


class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()
    # self.count = 0

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M -> ind: subvoxel index of an object, mask:
      cat (category id for peaks): B x M -> class id
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    # if self.count % 10 == 0:
    #     for c in range(3):
    #         neg_vis = neg_loss[0][c].view(234, 234).detach().cpu().numpy()
    #         plt.imshow(neg_vis, cmap='gray')
    #         plt.show()
        # gt_np = gt.max(dim=1)[0].view(234, 234).detach().cpu().numpy()
        # plt.imshow(gt_np, cmap='hot')
        # plt.show()
    # self.count += 1
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind)  # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos


class AttentionLoss(nn.Module):
    '''
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    '''
    def __init__(self):
        super(AttentionLoss, self).__init__()
        # self.alpha = 0.01
        # self.beta = 0.01
        # self.mha = nn.MultiheadAttention(11, 11) # (embed_dim, num_heads)
        self.attn_weight = nn.Conv1d(22, 11, 1)

    def dot_production_attention(self, x):
        attn_score = torch.matmul(x.transpose(-2, -1), x)
        attn_score = attn_score / math.sqrt(x.size(-2))
        attn_prob = F.softmax(attn_score, dim=-2)
        out = torch.matmul(x, attn_prob)
        return out

    def det_heads_attention(self, x):
        channel_attn = []
        head_attn = []
        # Self Attention per Channel      # self attention per channel
        channel_attn = self.dot_production_attention(x)

        # Self Attention per Heads
        pointer = 0
        for i in [3, 2, 1, 3, 2]:
            head_attn.append(self.dot_production_attention(x[:, :, pointer:pointer+i]))
            pointer += i
        head_attn = torch.cat(head_attn, dim=-1)

        combined_attn = torch.cat([channel_attn, head_attn], dim=-1)
        total_attn = self.attn_weight(combined_attn.transpose(-2, -1))

        return total_attn

    def forward(self, out, target, mask, ind):
        '''
        Arguments:
        out, target: B x C x H x W
        ind, mask: B x M -> ind: subvoxel index of an object, mask:
        cat (category id for peaks): B x M -> class id
        '''

        batch_size = out.size(0)
        pos_preds = _transpose_and_gather_feat(torch.cat([out, target], dim=0),
                                               torch.cat([ind, ind], dim=0))  # B x M x C
        s_out, t_out = pos_preds[:batch_size], pos_preds[batch_size:]
        mask = mask.float().unsqueeze(2)
        s_out *= mask
        t_out *= mask
        # s_out = s_out.permute(1, 0, 2)
        # pos_out, out_weight = self.mha(s_out, s_out, s_out)
        # t_out = t_out.permute(1, 0, 2)
        # pos_target, target_weight = self.mha(t_out, t_out, t_out)
        # value_loss = F.l1_loss(pos_out, pos_target, reduction='none')
        # weight_loss = F.l1_loss(out_weight, target_weight, reduction='none')
        # loss = (value_loss.transpose(0, 1).sum(dim=2).sum(dim=1) * self.alpha
        #         + weight_loss.sum(dim=2).sum(dim=1) * self.beta).sum()

        teacher_out = self.det_heads_attention(t_out)
        student_out = self.det_heads_attention(s_out)

        loss = F.l1_loss(student_out, teacher_out, reduction='none')
        loss = (loss / (mask.sum() + 1e-04)).sum()

        return loss


class HintonLoss(nn.Module):
    def __init__(self, lambda_=0.5, T=4.0):
        super(HintonLoss, self).__init__()
        self.lambda_ = lambda_
        self.T = T

    def calc_kl_div(self, teacher, student):
        return (teacher * (teacher / student)).sum(dim=2).sum(dim=1).mean()

    def forward(self, teacher, student, sigmoid=True):

        N, C, H, W = teacher.size()

        teacher_pred = teacher.permute(0, 2, 3, 1).reshape(N, -1, C)
        student_pred = student.permute(0, 2, 3, 1).reshape(N, -1, C)

        if sigmoid:
            teacher_pred = torch.sigmoid(teacher_pred)
            student_pred = torch.sigmoid(student_pred)
            return self.calc_kl_div(teacher_pred / self.T, student_pred / self.T) * (3e-02 * self.T ** 2)
        else:
            teacher_pred = F.softmax(teacher_pred / self.T, dim=1)
            student_pred = F.log_softmax(student_pred / self.T, dim=1)
            return F.kl_div(student_pred, teacher_pred, reduction='batchmean') * (self.T ** 2)


class IELoss(nn.Module):
    def __init__(self, lambda_=0.5):
        super(IELoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, i_t, i_s, e_t, e_s):

        N, C, H, W = i_t.size()
        i_t = torch.sigmoid(i_t).permute(0, 2, 3, 1).reshape(N, -1, C)
        i_s = torch.sigmoid(i_s).permute(0, 2, 3, 1).reshape(N, -1, C)
        e_t = torch.sigmoid(e_t).permute(0, 2, 3, 1).reshape(N, -1, C)
        e_s = torch.sigmoid(e_s).permute(0, 2, 3, 1).reshape(N, -1, C)

        i_kd = F.l1_loss(i_s, i_t)
        e_kd = F.l1_loss(e_s, e_t)
        ie = F.l1_loss(i_s, e_t)
        ei = F.l1_loss(e_s, i_t)

        pos = (i_kd + e_kd) * 0.5
        neg = (ie + ei) * 0.5

        return - torch.log(1 - pos) - torch.log(neg)   # min(pos) , max(neg)
        # return torch.log((i_kd + e_kd)) - torch.log(1 - (ie + ei))
        # return -(torch.log((i_kd + e_kd)/(H + W)) - torch.log(1 - (ie + ei)/(H + W))) -> 0.5331 mAP


class L2withSigmoid(nn.Module):
    def __init__(self, lambda_=0.5, T=4.0):
        super(L2withSigmoid, self).__init__()
        self.lambda_ = lambda_
        self.T = T

    def calc_kl_div(self, teacher, student):
        return (teacher * (teacher / student)).sum(dim=2).sum(dim=1).mean()

    def forward(self, teacher, student, sigmoid=True):

        N, C, H, W = teacher.size()

        teacher_pred = teacher.permute(0, 2, 3, 1).reshape(N, -1, C)
        student_pred = student.permute(0, 2, 3, 1).reshape(N, -1, C)

        if sigmoid:
            teacher_pred = torch.sigmoid(teacher_pred)
            student_pred = torch.sigmoid(student_pred)
        return F.mse_loss(teacher_pred / self.T, student_pred / self.T) * (10 * self.T ** 2)


class FocalKLDiv(nn.Module):
    def __init__(self, lambda_=0.5, T=4.0, gamma=2.0):
        super(FocalKLDiv, self).__init__()
        self.lambda_ = lambda_
        self.T = T
        self.gamma = gamma

    def calc_kl_div(self, teacher, student):
        loss = (((1 - student)**self.gamma) * (teacher * (teacher / student))).sum(dim=2).sum(dim=1).mean()
        return loss

    def forward(self, teacher, student, sigmoid=False):

        N, C, H, W = teacher.size()

        teacher_pred = teacher.permute(0, 2, 3, 1).reshape(N, -1, C)
        student_pred = student.permute(0, 2, 3, 1).reshape(N, -1, C)

        teacher_pred = torch.sigmoid(teacher_pred)
        student_pred = torch.sigmoid(student_pred)
        return self.calc_kl_div(teacher_pred / self.T, student_pred / self.T) * (1e-01 * self.T ** 2)


class PartialHintonLoss(nn.Module):
    def __init__(self, lambda_=0.5, T=4.0):
        super(PartialHintonLoss, self).__init__()
        self.lambda_ = lambda_
        self.T = T

    def forward(self, teacher, student, ind, sigmoid=False, softmax=True):
        batch_size = teacher.size(0)
        features = torch.cat([teacher, student], dim=0)
        ind = torch.cat([ind, ind], dim=0)
        preds = _transpose_and_gather_feat(features, ind)  # 2B x M x C
        p_size = preds.size()
        teacher_pred = preds[:batch_size].view(batch_size, p_size[1], p_size[2])
        student_pred = preds[batch_size:].view(batch_size, p_size[1], p_size[2])

        if sigmoid:
            teacher_pred = torch.sigmoid(teacher_pred)
            student_pred = torch.sigmoid(student_pred)

        if softmax:
            teacher_pred = F.softmax(teacher_pred / self.T, dim=2)
            student_pred = F.log_softmax(student_pred / self.T, dim=2)

        loss = F.kl_div(F.log_softmax(student_pred / self.T, dim=2),
                        F.softmax(teacher_pred / self.T, dim=2),
                        reduction='batchmean') * (self.T ** 2)
        return loss


class FitNetLoss(nn.Module):
    def __init__(self, p=2.0, in_channel=None, out_channel=None):
        super(FitNetLoss, self).__init__()
        self.p = p
        if not (in_channel is None and out_channel is None):
            self.channel_extension = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        else:
            self.channel_extension = None

    def forward(self, f_t, f_s, norm=False):

        if not self.channel_extension is None:
            f_s = self.channel_extension(f_s)

        return F.mse_loss(f_s, f_t), f_s


class HeadLoss(nn.Module):
    def __init__(self):
        super(HeadLoss, self).__init__()

    def forward(self, f_t, f_s):
        return F.normalize((f_s - f_t).abs()).mean()


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, reduction="mean", loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_bbox = self.loss_weight * self.smooth_l1_loss(
            pred,
            target,
            beta=self.beta,
        )
        return loss_bbox

    def smooth_l1_loss(self, pred, target, beta=1.0):
        assert beta > 0
        assert pred.size() == target.size() and target.numel() > 0
        diff = torch.abs(pred - target)
        abs_diff_lt_1 = torch.le(diff, 1 / (3.0 ** 2)).type_as(diff)
        loss = abs_diff_lt_1 * 0.5 * torch.pow(diff * 3.0, 2) + (diff - 0.5 / (3.0 ** 2)) * \
               (1.0 - abs_diff_lt_1)
        # loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        return loss