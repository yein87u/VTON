import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import Scale, ConvBlock, ConvAct, PreNorm



class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ConvBlock(inner_nc, inner_nc)]
        res_upconv = [ConvBlock(outer_nc, outer_nc)]

        downrelu = nn.GELU()
        uprelu = nn.GELU()
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        

class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

