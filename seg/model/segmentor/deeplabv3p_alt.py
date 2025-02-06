from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from gvcore.utils.types import TDataList
from gvcore.model import GenericModule

from model.backbone import make_backbone
from model.head.aspp_alt import ASPPV3pAlt


# Define a Cutout class for feature maps
class Cutout(nn.Module):
    def __init__(self, mask_size, p=0.5):
        super(Cutout, self).__init__()
        self.mask_size = mask_size
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() > self.p:
            return x
        _, _, h, w = x.size()
        y = x.clone()

        mask_size_half = self.mask_size // 2
        cx = torch.randint(0, w, (1,)).item()
        cy = torch.randint(0, h, (1,)).item()

        y[:, :, max(0, cy - mask_size_half):min(h, cy + mask_size_half),
        max(0, cx - mask_size_half):min(w, cx + mask_size_half)] = 0
        return y


# Define a Gaussian Noise Injection class for feature maps
class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


# Define a MixStyle perturbation class
class MixStyle(nn.Module):
    def __init__(self, p=0.5):
        super(MixStyle, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x

        batch_size = x.size(0)
        perm = torch.randperm(batch_size).to(x.device)
        mean = x.mean([2, 3], keepdim=True)
        std = x.std([2, 3], keepdim=True)
        mean_perm = mean[perm]
        std_perm = std[perm]
        return std_perm * (x - mean) / (std + 1e-6) + mean_perm

class DeeplabV3pAlt(GenericModule):
    def __init__(self, cfg):
        super(DeeplabV3pAlt, self).__init__(cfg)

        # Components ========
        self.backbone = make_backbone(cfg.model.backbone)

        self.decoder = ASPPV3pAlt(cfg.model.aspp)

        self.dropout = nn.Dropout2d(cfg.model.dropout)
        self.gaussian_noise = GaussianNoise(std=0.1)  # Gaussian noise perturbation
        self.cutout = Cutout(mask_size=16, p=0.5)  # Cutout on feature maps
        self.mixstyle = MixStyle(p=0.5)  # MixStyle perturbation

        self.classifier = nn.Conv2d(self.decoder.inner_channels, cfg.model.num_classes, 1)
    def apply_feature_perturbation(self, feats):
        # Apply various perturbations
        feats = self.gaussian_noise(feats)  # Apply Gaussian Noise
        feats = self.cutout(feats)  # Apply Cutout
        feats = self.mixstyle(feats)  # Apply MixStyle
        feats = self.dropout(feats)  # Apply Dropout
        return feats

    def forward_generic(self, data_list: TDataList) -> torch.Tensor:
        imgs = torch.stack([data.img for data in data_list], dim=0)

        feat_list = self.backbone(imgs)
        feats = self.decoder(feats=feat_list[-1], lowlevel_feats=feat_list[0])

        feats = self.apply_feature_perturbation(feats)

        logits = self.classifier(feats)
        logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=True)

        return logits

    def forward_train(self, data_list: TDataList) -> Dict:
        labels = torch.stack([data.label for data in data_list], dim=0)
        logits = self.forward_generic(data_list)
        return self.get_losses(logits, labels)

    def get_losses(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict:
        loss_dict = Dict()
        loss = F.cross_entropy(logits, labels.long().squeeze(1), ignore_index=255)

        loss_dict.loss = loss

        return loss_dict

    def forward_eval(self, data_list: TDataList):
        logits = self.forward_generic(data_list)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return Dict(pred_list=preds)

