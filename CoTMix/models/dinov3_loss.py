import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityLoss(nn.Module):
    """-H(p̄): âm entropy của phân phối dự đoán trung bình trên batch.

    Giá trị nhỏ nhất là -ln(K) khi dự đoán phân bố đều trên K lớp; tối thiểu hóa
    loss này ngăn mô hình dồn mọi mẫu đích vào 1-2 lớp.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits):
        mean_prob = F.softmax(logits, dim=1).mean(dim=0)
        return (mean_prob * torch.log(mean_prob + self.eps)).sum()


class DINOv3LossAddons(nn.Module):
    def __init__(self, src_mix_cls_weight=0.0, trg_div_weight=0.0, trg_cont_warmup_epochs=0):
        super().__init__()
        self.src_mix_cls_weight = float(src_mix_cls_weight)
        self.trg_div_weight = float(trg_div_weight)
        self.trg_cont_warmup_epochs = int(trg_cont_warmup_epochs)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.diversity_loss = DiversityLoss()

    @classmethod
    def from_hparams(cls, hparams):
        def get(key):
            return hparams[key] if key in hparams else 0
        return cls(get("src_mix_cls_weight"), get("trg_div_weight"), get("trg_cont_warmup_epochs"))

    def trg_cont_scale(self, epoch):
        #Hệ số nhân cho target contrastive loss: 0 trong warmup, 1 sau đó (epoch đếm từ 1)
        return 0.0 if epoch <= self.trg_cont_warmup_epochs else 1.0

    def forward(self, src_dominant_logits, src_y, trg_orig_logits):
        #Trả về (tổng loss có trọng số, dict giá trị từng loss để ghi log)
        total = src_dominant_logits.new_zeros(())
        logs = {}
        if self.src_mix_cls_weight > 0:
            src_mix_cls_loss = self.cross_entropy(src_dominant_logits, src_y)
            total = total + self.src_mix_cls_weight * src_mix_cls_loss
            logs["src_mix_cls_loss"] = src_mix_cls_loss.item()
        if self.trg_div_weight > 0:
            trg_div_loss = self.diversity_loss(trg_orig_logits)
            total = total + self.trg_div_weight * trg_div_loss
            logs["trg_div_loss"] = trg_div_loss.item()
        return total, logs
