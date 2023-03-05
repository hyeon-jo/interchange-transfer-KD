import torch
import torch.nn as nn

from ..registry import DETECTORS
from .. import builder
from det3d.models.losses.centernet_loss import *


class TSEncoder(nn.Module):
    def __init__(self, teacher_in, student_in, out_channel=32):
        super(TSEncoder, self).__init__()
        self.ds_teacher = nn.Conv2d(teacher_in, student_in, 1)
        self.us_student = nn.Conv2d(student_in, teacher_in, 1)

        self.downsample = nn.Sequential(
            nn.Conv2d(teacher_in, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, out_channel, 1)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(out_channel, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, student_in, 1)
        )
        self.criterion = SSLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, t, s, example=None, visual=False):
        s_in = self.us_student(s)
        t_encode = self.downsample(t)
        s_encode = self.downsample(s_in)
        t_decode = self.upsample(t_encode)
        s_decode = self.upsample(s_encode)
        s_out = self.us_student(s_decode)
        # t_out = self.us_student(t_decode)
        # return s_encode, t_encode
        if visual:
            return t_encode, s_encode, t_decode, s_decode, s_out

        s2t_loss = self.l1_loss(s_out, t) + self.criterion(s_out, example['mask'][0], example['ind'][0], t)
        t2s_loss = self.l1_loss(t_decode, s) + self.criterion(t_decode, example['mask'][0], example['ind'][0], s)
        encode_loss = self.l1_loss(s_encode, t_encode)

        return t2s_loss+s2t_loss+encode_loss


@DETECTORS.register_module
class KDNet(nn.Module):
    def __init__(
        self,
        teacher,
        student,
        student2=None,
        train_cfg=None,
        test_cfg=None,
        _lambda = 0.5,
        p = 2.0,
        distill_head=False,
        transformer=False,
    ):
        super(KDNet, self).__init__()
        # if not train_cfg is None:
        self.teacher = builder.build_detector(teacher, train_cfg=train_cfg, test_cfg=test_cfg)
        self.teacher.freeze()
        self.student = builder.build_detector(student, train_cfg=train_cfg, test_cfg=test_cfg)
        if not student2 is None:
            self.student2 = builder.build_detector(student2, train_cfg=train_cfg, test_cfg=test_cfg)
            self.student2.freeze()

        self._lambda = _lambda
        self.tse = TSEncoder(128*3, 32*3, out_channel=32)
        self.attn_loss = AttentionLoss()

    def forward(self, example, return_loss=True, elapse_time=None):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )
        student_feat, student_reader = self.student.extract_feat(data, ret_in_feature=True)
        student_pred, student_shared = self.student.bbox_head(student_feat)

        if return_loss:
            self.teacher.eval()
            with torch.no_grad():
                teacher_feat, _ = self.teacher.extract_feat(data)
                teacher_pred, teacher_shared = self.teacher.bbox_head(teacher_feat)

            losses = self.student.bbox_head.loss(example, student_pred)
            kd_losses = [float(0) for _ in range(len(student_pred))]
            student_box = torch.cat((student_pred[0]['hm'], student_pred[0]['reg'], student_pred[0]['height'],
                                     student_pred[0]['dim'], student_pred[0]['rot']), dim=1)
            teacher_box = torch.cat((teacher_pred[0]['hm'], teacher_pred[0]['reg'], teacher_pred[0]['height'],
                                     teacher_pred[0]['dim'], teacher_pred[0]['rot']), dim=1)

            loss = self.tse(teacher_feat, student_feat)
            for task_id in range(len(student_pred)):
                kd_losses[task_id] += loss

            for task_id in range(len(student_pred)):
                kd_losses[task_id] += self.attn_loss(student_box, teacher_box, example['mask'][0], example['ind'][0])

            for i in range(len(losses['loss'])):
                losses['loss'][i] = self._lambda * kd_losses[i] + (1 - self._lambda) * losses['loss'][i]
            losses['kd_losses'] = kd_losses

            return losses
        else:
            return self.student.bbox_head.predict(example, student_pred, self.student.test_cfg)
