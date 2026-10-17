from torch import nn


def get_backbone_class(backbone_name):
    """Return the feature-extractor class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Backbone not found: {}".format(backbone_name))
    return globals()[backbone_name]


class CNN(nn.Module):
    """Original CoTMix 1D-CNN encoder, retained as the baseline."""
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.feature_dim = configs.features_len * configs.final_out_channels

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        return x.reshape(x.shape[0], -1)


class DINOv3ViTS16(nn.Module):
    """Named wrapper so --backbone DINOv3ViTS16 works with CoTMix's factory."""
    def __init__(self, configs):
        super().__init__()
        # Import lazily: the original CNN baseline does not require Transformers.
        from models.dinov3 import DINOv3Encoder
        self.encoder = DINOv3Encoder(configs)
        self.feature_dim = self.encoder.feature_dim

    def forward(self, x):
        return self.encoder(x)


class classifier(nn.Module):
    def __init__(self, configs, input_dim=None):
        super(classifier, self).__init__()
        if input_dim is None:
            input_dim = configs.features_len * configs.final_out_channels
        self.logits = nn.Linear(input_dim, configs.num_classes)

    def forward(self, x):
        return self.logits(x)
