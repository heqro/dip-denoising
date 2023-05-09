import torch
from torch import nn

# adjust for using the last GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # define skip connections processing modules
        self.skips = nn.ModuleList()
        for channels_in in [32] + [128] * 4:
            self.skips.append(nn.Sequential(
                nn.Conv2d(channels_in, 4, kernel_size=(1, 1), device=dev),
                nn.BatchNorm2d(4, device=dev),
                nn.LeakyReLU(0.2)
            ))

        # define "double-convolution" downsampling modules
        self.double_convs = nn.ModuleList()
        for channels_in in [32] + [128] * 4:
            self.double_convs.append(nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(
                    channels_in, 128, kernel_size=(3, 3), stride=(2, 2), device=dev),
                nn.BatchNorm2d(128, device=dev),
                nn.LeakyReLU(0.2),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(
                    128, 128, kernel_size=(3, 3), device=dev),
                nn.BatchNorm2d(128, device=dev),
                nn.LeakyReLU(0.2),
            ))

        # define "post-concatenation" processing modules
        self.convs_after_concat = nn.ModuleList()
        for i in range(5):
            module = nn.Sequential(
                nn.BatchNorm2d(132, device=dev),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(132, 128, kernel_size=(3, 3), device=dev),
                nn.BatchNorm2d(128, device=dev),
                nn.LeakyReLU(0.2),
                # nn.ReflectionPad2d((0, 0, 0, 0)),
                nn.Conv2d(128, 128, kernel_size=(1, 1), device=dev),
                nn.BatchNorm2d(128, device=dev),
                nn.LeakyReLU(0.2)
            )
            if i != 0:  # deep layer
                module.append(nn.Upsample(scale_factor=2.0,
                              mode='bilinear'))
            else:  # last layer
                # module.append(nn.ReflectionPad2d((0, 0, 0, 0)))
                module.append(
                    nn.Conv2d(128, 3, kernel_size=(1, 1), device=dev))
                module.append(nn.Sigmoid())
            self.convs_after_concat.append(module)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        skip_z = self.skips[0](z)
        x1 = self.double_convs[0](z)
        skip_x1 = self.skips[1](x1)
        x2 = self.double_convs[1](x1)
        skip_x2 = self.skips[2](x2)
        x3 = self.double_convs[2](x2)
        skip_x3 = self.skips[3](x3)
        x4 = self.double_convs[3](x3)
        skip_x4 = self.skips[4](x4)
        x5 = self.double_convs[4](x4)

        y5 = self.convs_after_concat[4](torch.cat([skip_x4, nn.Upsample(scale_factor=2.0,
                                                                        mode='bilinear')(x5)], dim=1))
        y4 = self.convs_after_concat[3](torch.cat([skip_x3, y5], dim=1))
        y3 = self.convs_after_concat[2](torch.cat([skip_x2, y4], dim=1))
        y2 = self.convs_after_concat[1](torch.cat([skip_x1, y3], dim=1))
        y1 = self.convs_after_concat[0](torch.cat([skip_z, y2], dim=1))

        return y1

    # def forward(self, z: torch.Tensor):
    #     skips = []
    #     x_vec = []
    #     x = z
    #     for i in range(5):
    #         skips.append(self.skips[i](x))
    #         x = self.double_convs[i](x)
    #         x_vec.append(x)
    #
    #     _tmp = self.convs_after_concat[4](
    #         torch.cat([
    #             skips[4], nn.Upsample(scale_factor=2.0, mode='bilinear')(x_vec[4])],
    #             dim=1))
    #
    #     for i in reversed(range(4)):
    #         _tmp = self.convs_after_concat[i](
    #             torch.cat([skips[i], x_vec[i]], dim=1))
    #
    #     return _tmp


