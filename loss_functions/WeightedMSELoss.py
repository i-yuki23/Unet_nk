import torch
import torch.nn.functional as F
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, zero_weight=0):
        super(WeightedMSELoss, self).__init__()
        self.zero_weight = zero_weight

    def forward(self, y_pred, y_true):
        weight = torch.ones_like(y_true)
        weight[y_true == 0] = self.zero_weight

        loss = F.mse_loss(y_pred, y_true, reduction='none')  # 各ピクセルのMSEを取得
        loss = (loss * weight).mean()  # 重みを適用して平均をとる
        return loss
