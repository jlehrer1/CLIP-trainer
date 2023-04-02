import torch.nn as nn
import torch 
from typing import Optional, Union

class StandardResidualBlock(nn.Module):
    """Describes the residual block used in the shallower resnet architectures from the original paper. The 
    alternative is to use the Bottleneck block, which we implement below. Source: https://arxiv.org/abs/1512.03385."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.block = nn.Sequential(
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
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualBottleneckBlock(nn.Module):
    """Describes the bottleneck block used in the deeper resnet architectures from the original paper. """

    EXPANSION: int = 4

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels, 
                out_channels=self.out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0, 
                bias=self.bias
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels * ResidualBottleneckBlock.EXPANSION,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=self.bias
            ),
            nn.BatchNorm2d(self.out_channels * ResidualBottleneckBlock.EXPANSION),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.block(x)

        return x


class ResNetX(nn.Module):
    def __init__(
            self, 
            n_blocks: int, 
            in_channels: int, 
            out_channels: int, 
            embedding_dim: int, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False,
            block: Optional[Union[ResidualBottleneckBlock, StandardResidualBlock]] = None,
        ):
        super().__init__()
        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.block = block

        if self.block is None:
            self.block = StandardResidualBlock if self.n_blocks < 50 else ResidualBottleneckBlock

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   
        )

        self.layer_block = nn.Sequential(
            self._make_layer(self.in_channels, 64, self.n_blocks),
            self._make_layer(64 * self.block.EXPANSION, 128, self.n_blocks),
            self._make_layer(128 * self.block.EXPANSION, 256, self.n_blocks),
            self._make_layer(256 * self.block.EXPANSION, 512, self.n_blocks),
        )

        self.output_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),  # flatten all dimensions except batch
            nn.Linear(512 * self.block.EXPANSION, self.embedding_dim),
        )

        self._init_weights()

    def _make_layer(self, in_channels: int, out_channels: int, n_blocks: int) -> nn.Sequential:
        layers = []
        for _ in range(n_blocks):
            layers.append(self.block(in_channels, out_channels, self.kernel_size, self.stride, self.padding, self.bias))
            in_channels = out_channels * self.block.EXPANSION if self.block == ResidualBottleneckBlock else out_channels

        return nn.Sequential(*layers)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_block(x)
        x = self.layer_block(x)
        x = self.output_block(x)

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
