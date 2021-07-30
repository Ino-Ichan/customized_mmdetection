from typing import Sequence, Tuple, Optional

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES

import timm


@BACKBONES.register_module()
class TimmBB(nn.Module):
    def __init__(
            self,
            name: str,
            pretrained: Optional[str] = "imagenet",
            out_indices: Sequence[int] = (1, 2, 3, 4),
            frozen_stages: int = -1,
            norm_eval: bool = True,
            sync_bn: bool = False,
            **kwargs):
        super().__init__()

        print(pretrained)
        self.net = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            **kwargs
        )
        if sync_bn:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self._freeze_stages()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        feats_all = self.net(x)
        return tuple(feats_all)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_stages(self):
        pass

    def init_weights(self, **_):
        pass