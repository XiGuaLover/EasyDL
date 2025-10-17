import torch
import torch.nn as nn
from utils.ConfigType import PredFormerComponentConfig

from ..components.PredFormerComponent import PredFormer_Model


class PredFormer(nn.Module):
    def __init__(
        self,
        componentConfig: PredFormerComponentConfig,
    ):
        super().__init__()
        self.model = PredFormer_Model(componentConfig)

    def forward(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        inputSeqLen = batch_x.shape[1]
        targetSeqLen = batch_y.shape[1]

        """Forward the model"""
        if targetSeqLen == inputSeqLen:
            pred_y = self.model(batch_x)
        elif targetSeqLen < inputSeqLen:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :targetSeqLen]
        elif targetSeqLen > inputSeqLen:
            pred_y = []
            d = targetSeqLen // inputSeqLen
            m = targetSeqLen % inputSeqLen

            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])

            pred_y = torch.cat(pred_y, dim=1)

        return pred_y
