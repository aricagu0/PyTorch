import torch
import torch.nn as nn

class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.BatchNorm2d(36), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=3, stride=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self._fc_input = self._get_fc_input()
        self.fc = nn.Sequential(
            nn.Linear(self._fc_input, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 16),   nn.ReLU(),
            nn.Linear(16, 1),    nn.Tanh()
        )

    def _get_fc_input(self):
        dummy = torch.zeros(1, 1, 66, 200)
        out = self.flatten(self.features(dummy))
        return out.shape[1]

    def forward(self, x):
        return self.fc(self.flatten(self.features(x))).squeeze(1)

if __name__ == '__main__':
    m = PilotNet()
    dummy = torch.zeros(4, 1, 66, 200)
    out = m(dummy)
    print(f'모델 출력 shape: {out.shape}')
    total = sum(p.numel() for p in m.parameters())
    print(f'총 파라미터 수: {total:,}')
