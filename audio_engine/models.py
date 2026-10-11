"""Models (a): CNN-custom + CNN+Transformer from scratch @2000Hz/2s.

Replaces MIT/ast-finetuned-audioset (16kHz-locked, PEFT+ONNX friction).
Same Mel 64x128 input -> fair benchmark. Pure torch -> ONNX-clean.
"""
import torch
import torch.nn as nn


class CNNFront(nn.Module):
    def __init__(self, c=(1, 32, 64, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c[0], c[1], 3, padding=1), nn.BatchNorm2d(c[1]), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c[1], c[2], 3, padding=1), nn.BatchNorm2d(c[2]), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c[2], c[3], 3, padding=1), nn.BatchNorm2d(c[3]), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 32)),  # [B,128,4,32]
        )

    def forward(self, x):
        return self.net(x)


class CNNCustom(nn.Module):
    """Model #3: 3-conv baseline (~200k params)."""

    def __init__(self, n_cls=2):
        super().__init__()
        self.front = CNNFront()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(128 * 4 * 32, 128), nn.ReLU(),
                                  nn.Linear(128, n_cls))

    def forward(self, x):
        return self.head(self.front(x))


class CNNTransformer(nn.Module):
    """Model #5 (a): CNN frontend + 2x TransformerEncoder for B1/B2 time structure."""

    def __init__(self, d=128, heads=4, layers=2, n_cls=2):
        super().__init__()
        self.front = CNNFront()  # [B,128,4,32] -> treat 32 as time, 4*128 as feat
        self.proj = nn.Linear(128 * 4, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=256,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=layers)
        self.pos = nn.Parameter(torch.randn(1, 32, d) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_cls))

    def forward(self, x):
        f = self.front(x)                       # [B,128,4,32]
        f = f.permute(0, 3, 1, 2).flatten(2)    # [B,32,512]
        t = self.proj(f) + self.pos             # [B,32,d]
        t = self.tr(t).mean(dim=1)              # mean-pool time
        return self.head(t)


def get(name, **kw):
    return {"cnn": CNNCustom, "cnn_transformer": CNNTransformer}[name](**kw)


if __name__ == "__main__":  # ponytail: runnable self-check
    for n in ("cnn", "cnn_transformer"):
        m = get(n).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 1, 64, 128))
        assert out.shape == (2, 2), (n, out.shape)
        print("ok", n, sum(p.numel() for p in m.parameters()), "params")
