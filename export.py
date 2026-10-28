import torch
import sys
import wandb

# Vá lỗi đệ quy của WandB một cách an toàn (chỉ can thiệp hàm __getattr__)
def patched_getattr(self, key):
    if key == "_items":
        raise AttributeError("Ngăn chặn đệ quy sớm")
    try:
        return self.__getitem__(key)
    except KeyError:
        raise AttributeError(key)

# Gắn bản vá vào class gốc thay vì xóa nó đi
wandb.sdk.wandb_config.Config.__getattr__ = patched_getattr
sys.setrecursionlimit(50000)

print("Đang tải mô hình pkl cũ...")
model = torch.load("cotmix_uci_har.pkl", map_location="cpu", weights_only=False)
model.eval()

class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_extractor = model.feature_extractor
        self.classifier = model.classifier
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

wrapper = InferenceWrapper(model)
wrapper.eval()

print("Đang biên dịch sang TorchScript...")
dummy_input = torch.randn(1, 9, 128)
traced_model = torch.jit.trace(wrapper, dummy_input)

traced_model.save("cotmix_uci_har.pt")
print("Đã tạo thành công tệp cotmix_uci_har.pt chuẩn!")