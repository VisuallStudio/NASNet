import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal
from networks.submodules import build_corr, ResBlock

class SharedLayer(nn.Module):
    
    def __init__(self, input_channel=3, output_channel=32, kernel_size=3, block=nn.Conv2d, depth=1):
        super(SharedLayer, self).__init__()

        if depth == 1:
            self.block = block(input_channel, output_channel, kernel_size=kernel_size, stride=2)
        else:
            self.block = nn.Sequential(
                block(input_channel, output_channel, kernel_size=kernel_size, stride=2),
                block(output_channel, output_channel, kernel_size=kernel_size, stride=1)
            )

    def forward(self, left_fea, right_fea):

        block_l = self.block(left_fea)
        block_r = self.block(right_fea)

        return block_l, block_r, None

class MatchLayer(nn.Module):

    def __init__(self, maxdisp=192, input_channel=3, output_channel=32):
        super(MatchLayer, self).__init__()
        
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.maxdisp = maxdisp

        self.corr_width = maxdisp

        # shrink and extract features
        self.block1 = ResBlock(input_channel, output_channel, stride=2)
        self.block2 = ResBlock(output_channel, output_channel, stride=1)

        self.corr_act = nn.LeakyReLU(0.1, inplace=True)
        # self.left_block = ResBlock(output_channel, output_channel//4, stride=1)

        self.fusion_block = ResBlock(maxdisp+output_channel, output_channel, stride=1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, left_fea, right_fea):

        # split left image and right image
        block_l = self.block1(left_fea)
        block_l = self.block2(block_l)
        block_r = self.block1(right_fea)
        block_r = self.block2(block_r)

        corr_fea = build_corr(block_l, block_r, max_disp=self.maxdisp)
        corr_fea = self.corr_act(corr_fea)
        concat_corr = torch.cat((block_l, corr_fea), 1)
        out_corr = self.fusion_block(concat_corr)

        return block_l, block_r, out_corr

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class FeatureLayer(nn.Module):

    def __init__(self, input_channel=3, output_channel=32, kernel_size=3, block=nn.Conv2d, depth=1):
        super(FeatureLayer, self).__init__()
        
        self.input_channel = input_channel
        self.output_channel = output_channel

        if depth == 1:
            self.block = block(input_channel, output_channel, kernel_size=kernel_size, stride=2)
        else:
            self.block = nn.Sequential(
                block(input_channel, output_channel, kernel_size=kernel_size, stride=2),
                block(output_channel, output_channel, kernel_size=kernel_size, stride=1)
            )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, features):

        # extract high-level features
        return self.block(features)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class UpSamplingLayer(nn.Module):

    def __init__(self, input_channel, output_channel=16, inter_channel=None):
        super(UpSamplingLayer, self).__init__()
        
        self.input_channel = input_channel
        self.output_channel = output_channel

        if inter_channel is None:
            self.inter_channel = input_channel + 1
        else:
            self.inter_channel = inter_channel
        # self.block1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1,inplace=True)
        )
        self.block2 = nn.ConvTranspose2d(self.inter_channel, output_channel, kernel_size=3, stride=1, padding=1)

        self.disp_up = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.disp_regr = nn.Conv2d(input_channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, bottom_fea1, bottom_fea2, coarse_disp=None):

        disp = self.disp_regr(bottom_fea1)
        if coarse_disp is not None:
            disp += coarse_disp

        block1 = self.block1(bottom_fea1)
        upsampled_disp = self.disp_up(disp)
        concat_fea = torch.cat((block1, upsampled_disp, bottom_fea2), 1)
        block2 = self.block2(concat_fea)

        return block2, disp

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


