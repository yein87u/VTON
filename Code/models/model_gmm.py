import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import ConvBlock, ConvAct, PreNorm


def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])    
    grid_list = reversed(grid_list)
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...] for dim, grid in enumerate(grid_list)] 
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0 for grid, size in zip(grid_list, reversed(sizes))] 
    return torch.stack(grid_list, dim=-1) 


class FlowEstimator(nn.Module):
    def __init__(self, dim_in, dim_out=2, dim_mid=128, norm_type='BN'):
        super().__init__()
        self.module = torch.nn.Sequential(
            ConvAct(dim_in , dim_mid, kernel_size=1),
            ConvAct(dim_mid, dim_mid, kernel_size=3, groups=dim_mid),
            ConvAct(dim_mid, dim_mid, kernel_size=3, groups=dim_mid),
            ConvAct(dim_mid, dim_out, kernel_size=1, use_actv=False),
        )
        self.module = PreNorm(dim_in, self.module, norm_type)
   
    def forward(self, wapring_fea, shape_fea):
        concat = torch.cat([wapring_fea, shape_fea], dim=1)
        return self.module(concat) 
    

class CascadeWarpingModule(nn.Module):
    def __init__(self, dim_in, norm_type):
        super().__init__()
        self.shapeNet   = FlowEstimator(dim_in * 2, norm_type=norm_type)
        self.textureNet = FlowEstimator(dim_in * 2, norm_type=norm_type)
    
    def forward(self, shape_fea, cloth_fea, shape_last_flow=None, cloth_last_flow=None):
        # Coarse
        if shape_last_flow is not None:
            cloth_fea_ = F.grid_sample(cloth_fea, shape_last_flow.permute(0, 2, 3, 1).detach(), mode='bilinear', padding_mode='border')
        else:
            cloth_fea_ = cloth_fea

        shape_delta_flow = self.shapeNet(cloth_fea_, shape_fea)
        shape_flow = apply_offset(shape_delta_flow)

        if shape_last_flow is not None:
            shape_last_flow = F.grid_sample(shape_last_flow, shape_flow, mode='bilinear', padding_mode='border')
        else:
            shape_last_flow = shape_flow.permute(0, 3, 1, 2)    # b, c, h, w

        # Fine
        if cloth_last_flow is not None:
            cloth_last_flow_ = F.grid_sample(shape_last_flow, cloth_last_flow.permute(0, 2, 3, 1).detach(), mode='bilinear', padding_mode='border')
        else:
            cloth_last_flow_ = shape_last_flow.clone()
        cloth_fea_ = F.grid_sample(cloth_fea, cloth_last_flow_.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

        cloth_delta_flow = self.textureNet(cloth_fea_, shape_fea)
        cloth_flow = apply_offset(cloth_delta_flow)             # b, h, w, c

        if cloth_last_flow is not None:
            cloth_last_flow = F.grid_sample(cloth_last_flow, cloth_flow, mode='bilinear', padding_mode='border')
        else:
            cloth_last_flow = cloth_flow.permute(0, 3, 1, 2)    # b, c, h, w
        
        cloth_last_flow_ = F.grid_sample(shape_last_flow, cloth_last_flow.permute(0, 2, 3, 1), mode="bilinear", padding_mode='border')
        # Upsample
        cloth_last_flow_ = F.interpolate(cloth_last_flow_, scale_factor=2, mode='bilinear')
        cloth_last_flow = F.interpolate(cloth_last_flow, scale_factor=2, mode='bilinear')
        shape_last_flow = F.interpolate(shape_last_flow, scale_factor=2, mode='bilinear')
        return shape_last_flow, cloth_last_flow, cloth_last_flow_, shape_delta_flow, cloth_delta_flow
    
#堆疊多個卷積層(ConvBlock)來提取輸入資料的特徵，為深度學習的骨幹
class Backbone(nn.Module):
    #         資料的通道數(維度),定義每一層卷積層的輸出通道數,指定使用的正規化層類型(BN:批次正規化)
    def __init__(self, dim_in, channels, norm_type='BN'):
        super().__init__()
        self.stage1 = ConvBlock(dim_in     , channels[0], stride=2, norm_type=norm_type)
        self.stage2 = ConvBlock(channels[0], channels[1], stride=2, norm_type=norm_type)
        self.stage3 = ConvBlock(channels[1], channels[2], stride=2, norm_type=norm_type)
        self.stage4 = ConvBlock(channels[2], channels[3], stride=2, norm_type=norm_type)
        
    def forward(self, x):
        out1 = self.stage1(x)           # x (64 , 128, 96)
        out2 = self.stage2(out1)        # x (96 , 64 , 48)
        out3 = self.stage3(out2)        # x (128, 32 , 24) 
        out4 = self.stage4(out3)        # x (256, 16 , 12) 
        return [out1, out2, out3, out4]
    

class FPN(nn.Module):
    def __init__(self, dim_ins, dim_out=256):
        super().__init__()
        # adaptive 
        self.adaptive = []
        for in_chns in list(reversed(dim_ins)):
            adaptive_layer = nn.Conv2d(in_chns, dim_out, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(dim_ins)):
            smooth_layer = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=dim_out)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x
        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)
        return tuple(reversed(feature_list))


class AFlowNet(nn.Module):
    def __init__(self, channels, norm_type='BN'):
        super().__init__()
        self.num_layers = len(channels)
        self.flow = nn.ModuleList([CascadeWarpingModule(channels[-1], norm_type) for _ in range(self.num_layers)]) 

    def spatial_transform(self, to_warping, flow, padding_mode='border'):
        return F.grid_sample(to_warping, flow.permute(0, 2, 3, 1), mode="bilinear", padding_mode=padding_mode)
    
    def forward(self, cloth, cloth_mask, shape_list, cloth_list):
        warping_masks = []
        warping_cloths = []
        shape_last_flows = []
        cloth_last_flows = []
        shape_delta_flows = []
        cloth_delta_flows = []
        for i in range(self.num_layers):
            shape_fea = shape_list[-(i+1)]
            cloth_fea = cloth_list[-(i+1)]

            if i == 0:
                shape_last_flow, cloth_last_flow, cloth_last_flow_, shape_delta_flow, cloth_delta_flow = self.flow[i](shape_fea, cloth_fea)
            else:
                shape_last_flow, cloth_last_flow, cloth_last_flow_, shape_delta_flow, cloth_delta_flow = self.flow[i](shape_fea, cloth_fea, shape_last_flow, cloth_last_flow)

            _, _, h, w = shape_last_flow.shape
            
            cloth_ = F.interpolate(cloth, size=(h, w), mode='bilinear')
            cloth_mask_ = F.interpolate(cloth_mask, size=(h, w), mode='nearest')

            cloth_ = self.spatial_transform(cloth_, cloth_last_flow_, padding_mode='border')
            cloth_mask_ = self.spatial_transform(cloth_mask_, shape_last_flow, padding_mode='zeros')
            
            warping_cloths.append(cloth_)
            warping_masks.append(cloth_mask_)
            shape_last_flows.append(shape_last_flow)
            cloth_last_flows.append(cloth_last_flow_)
            shape_delta_flows.append(shape_delta_flow)
            cloth_delta_flows.append(cloth_delta_flow)

        return {
            'warping_masks':warping_masks,
            'warping_cloths':warping_cloths,
            'shape_last_flows': shape_last_flows, 
            'cloth_last_flows': cloth_last_flows, 
            'shape_delta_flows': shape_delta_flows, 
            'cloth_delta_flows': cloth_delta_flows, 
        }


class CAFWM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.channels = [64, 96, 128, 128]  #設定通道

        self.backbone_A = Backbone(3 , self.channels, norm_type='BN')
        self.backbone_B = Backbone(23, self.channels, norm_type='BN')
        self.FPN_A = FPN(self.channels, dim_out=self.channels[-1])
        self.FPN_B = FPN(self.channels, dim_out=self.channels[-1])
        self.dec_tryon = AFlowNet(self.channels, norm_type='BN')
    
    def forward(self, cloth, cloth_mask, person_shape):
        cloth_list = self.FPN_A(self.backbone_A(cloth)) 
        shape_list = self.FPN_B(self.backbone_B(person_shape))
        output = self.dec_tryon(cloth, cloth_mask, shape_list, cloth_list)
        return output