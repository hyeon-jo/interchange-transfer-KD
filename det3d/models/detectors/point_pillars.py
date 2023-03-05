from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn
from .. import builder
from copy import deepcopy
import torch.nn.functional as F
import torch
from ..utils.finetune_utils import FrozenBatchNorm2d
import matplotlib.pyplot as plt

@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        autoencoder=False,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def freeze_module(self, reader=False, neck=False, bbox_head=False):
        if reader:
            for p in self.reader.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.reader)

        if neck:
            for p in self.neck.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.neck)

        if bbox_head:
            for p in self.bbox_head.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.bbox_head)

        return self

    def extract_feat(self, data, ret_in_feature=True):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        input_features = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        x = input_features
        if self.with_neck:
            x = self.neck(x)
        if ret_in_feature:
            return x, input_features
        else:
            return x, None

    def forward(self, example, return_loss=True, **kwargs):
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

        x, _ = self.extract_feat(data)
        preds, _ = self.bbox_head(x)
        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
