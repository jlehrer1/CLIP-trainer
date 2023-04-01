import torch.nn as nn
import torch 

class ResNetX(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, embedding_dim, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels, 
                    out_channels=self.out_channels, 
                    kernel_size=self.kernel_size, 
                    stride=self.stride, 
                    padding=self.padding, 
                    bias=self.bias
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(),
            ) for _ in range(self.n_blocks)]
        )

        self.ff = nn.Linear(self.out_channels, self.embedding_dim)
        self._init_weights()

    def forward(self, x):
        print(x)
        for block in self.blocks:
            x = x + block(x)

        x = torch.flatten(x, 1)
        x = self.ff(x)

        return x

    def _init_weights(self):
        # defines typical weight initialization for resnet 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
