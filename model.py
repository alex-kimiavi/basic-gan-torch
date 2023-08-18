import torch
import torch.nn as nn
import torch.nn.functional as F



class GenLayer(nn.Module):

    def __init__(self, in_filter, out_filter, kernel_size, stride, padding, bias=False):

        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_filter, out_filter, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_filter),
            nn.ReLU()
        )
        nn.init.normal_(self.layer[0].weight.data, 0.0, 0.02)
        nn.init.normal_(self.layer[1].weight.data, 1.0, 0.02)
        nn.init.normal_(self.layer[1].bias.data, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class Generator(nn.Module):

    def __init__(self):

        super().__init__()
        self.t1 = GenLayer(100, 1024, 4, 1, 0)
        self.t2 = GenLayer(1024, 512, 4, 2, 1)
        self.t3 = GenLayer(512, 256, 4, 2, 1)
        self.t4 = GenLayer(256, 128, 4, 2, 1)
        self.t5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t1(x) 
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = F.tanh(self.t5(x))
        return x

class DiscLayer(nn.Module):

    def __init__(self, in_filter, out_filter, k, s, p, b=False):

        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_filter, out_filter, k, s, p, bias=b),
            nn.BatchNorm2d(out_filter),
            nn.LeakyReLU(inplace=True)
        )
        nn.init.normal_(self.layer[0].weight.data, 0.0, 0.02)
        nn.init.normal_(self.layer[1].weight.data, 1.0, 0.02)
        nn.init.normal_(self.layer[1].bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class Discriminator(nn.Module):

    def __init__(self):
        
        super().__init__()
        self.c1 = DiscLayer(3, 32, 4, 2, 1)
        self.c2 = DiscLayer(32, 64, 4, 2, 1)
        self.c3 = DiscLayer(64, 128, 4, 2, 1)
        self.c4 = DiscLayer(128, 256, 4, 2, 1)
        self.c5 = nn.Conv2d(256, 1, 4, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = F.sigmoid(self.c5(x))
        return x
    
