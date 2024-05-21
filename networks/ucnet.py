import copy
import random
import sys

import math
import numpy
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal

from networks.modules import MatchLayer, UpSamplingLayer, FeatureLayer
from networks.submodules import warp_right_to_left, channel_length, ResBlock, build_corr
from ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer, DynamicMBConvLayerTest
from ofa.utils.layers import IdentityLayer, ResidualBlock
from ofa.utils.my_modules import MyNetwork, set_bn_param, get_bn_param

from utils.common import val2list, make_divisible


class UCRefineNet(nn.Module):

    def __init__(self, scale=6, init_channel=32):

        super(UCRefineNet, self).__init__()

        self.scale = scale
        self.init_channel = init_channel

        self.down_layers = []
        self.up_layers = []
        for i in range(scale):
            input_channel = init_channel * (2 ** (i - 1))
            output_channel = init_channel * (2 ** (i))
            up_input_channel = output_channel
            up_output_channel = input_channel
            if i == 0:
                # concat img0(3), img1(3), img1->img0(3), flow(1), diff-img(1)
                self.down_layers.append(
                    FeatureLayer(input_channel=11, output_channel=init_channel, kernel_size=3, block=ResBlock, depth=2))
            else:
                self.down_layers.append(
                    FeatureLayer(input_channel=input_channel, output_channel=output_channel, block=ResBlock, depth=2))
            if i == 0:
                self.up_layers.append(UpSamplingLayer(input_channel=init_channel, output_channel=init_channel,
                                                      inter_channel=init_channel + 3 + 1))
            else:
                self.up_layers.append(UpSamplingLayer(input_channel=up_input_channel, output_channel=up_output_channel))

        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)

        self.last_disp_regr = nn.Conv2d(init_channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, inputs, disps):

        down_feas = []
        up_feas = []
        final_disps = []

        for i in range(self.scale):
            if i == 0:
                down_fea = self.down_layers[i](inputs)
            else:
                down_fea = self.down_layers[i](down_feas[i - 1])
            down_feas.append(down_fea)

        for i in range(self.scale, 0, -1):
            if i == self.scale:
                up_fea, disp = self.up_layers[i - 1](down_feas[i - 1], down_feas[i - 2], disps[self.scale - i])
            elif i > 1:
                up_fea, disp = self.up_layers[i - 1](up_feas[-1], down_feas[i - 2], disps[self.scale - i])
            else:
                left_img = inputs[:, :3, :, :]
                up_fea, disp = self.up_layers[i - 1](up_feas[-1], left_img, disps[self.scale - i])

            up_feas.append(up_fea)
            final_disps.append(disp)

        last_res_disp = self.last_disp_regr(up_feas[-1])
        final_disp = disps[-1] + last_res_disp
        final_disps.append(final_disp)
        final_disps = [self.relu(final_disp) for final_disp in final_disps]

        return final_disps


class OFAUCRefineNet(nn.Module):
    def __init__(self, init_channel=32, width_mult=1.0,
                 ks_list=7, expand_ratio_list=8, depth_list=4, scale_list=6):
        super(OFAUCRefineNet, self).__init__()
        self.init_channel = init_channel
        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.scale_list = val2list(scale_list, 1)
        self.max_scale = max(self.scale_list)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()
        self.scale_list.sort()

        # self.feature_blocks_down = nn.ModuleList(self.ConstructFeatureNet_down())
        self.feature_blocks_down = nn.ModuleList(self.Test_down())
        # self.feature_runtime_depth_down = [len(block_idx) for block_idx in self.fea_block_group_info_down]
        # self.feature_blocks_up = nn.ModuleList(self.ConstructFeatureNet_up())
        self.feature_blocks_up = nn.ModuleList(self.Test_up())
        # self.feature_blocks_up = self.is_decoder_ofa()
        # self.feature_runtime_depth_up = [len(block_idx) for block_idx in self.fea_block_group_info_up]

        self.last_disp_regr = nn.Conv2d(init_channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def is_decoder_ofa(self):
        up_layers = []
        for i in range(max(self.scale_list)):
            input_channel = self.init_channel * (2 ** (i - 1))
            output_channel = self.init_channel * (2 ** (i))
            up_input_channel = output_channel
            up_output_channel = input_channel

            if i == 0:
                up_layers.append(UpSamplingLayer(input_channel=self.init_channel, output_channel=self.init_channel,
                                                 inter_channel=self.init_channel + 3 + 1))
            else:
                up_layers.append(
                    UpSamplingLayer(input_channel=up_input_channel, output_channel=up_output_channel))
        up_layers.reverse()
        up_layers = nn.ModuleList(up_layers)
        return up_layers

    def Test_down(self):
        base_stage_width = [self.init_channel] + [self.init_channel * (2 ** i) for i in range(1, max(self.scale_list))]
        stride_stages = [2] + [2 for _ in range(max(self.scale_list) - 1)]
        act_stages = ['relu' for _ in range(max(self.scale_list))]  # relu
        se_stages = [False for _ in range(self.max_scale)]
        width_list = []
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)
        # inverted residual blocks
        blocks = []
        _block_index = 0
        feature_dim = 11
        for width, s, act_func, use_se in zip(width_list, stride_stages, act_stages, se_stages):
            output_channel = width
            mobile_inverted_conv = DynamicMBConvLayerTest(
                in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                stride=s, act_func=act_func, use_se=use_se, is_transposeConv=False, depth_list=max(self.depth_list)
            )
            if s == 1 and feature_dim == output_channel:
                shortcut = IdentityLayer(feature_dim, feature_dim)
            else:
                shortcut = None
            blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
            feature_dim = output_channel
        return blocks

    def Feature_extraction_Test_down(self, features):
        x = features
        down_features = []
        for i in range(len(self.feature_blocks_down)):
            x = self.feature_blocks_down[i](x)
            down_features.append(x)
        return down_features

    def ConstructFeatureNet_down(self):
        base_stage_width = [self.init_channel] + [self.init_channel * (2 ** i) for i in range(1, max(self.scale_list))]
        stride_stages = [2] + [2 for _ in range(max(self.scale_list) - 1)]
        act_stages = ['relu' for _ in range(max(self.scale_list))]  # relu
        # se_stages = [False if i % 2 == 0 else True for i in range(self.max_scale)]
        se_stages = [False for _ in range(self.max_scale)]
        n_block_list = [max(self.depth_list)] * max(self.scale_list)
        width_list = []
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)
        # inverted residual blocks
        self.fea_block_group_info_down = []
        blocks = []
        _block_index = 0
        feature_dim = 11
        for width, n_block, s, act_func, use_se in zip(width_list, n_block_list,
                                                       stride_stages, act_stages, se_stages):
            self.fea_block_group_info_down.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                    kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                    stride=stride, act_func=act_func, use_se=use_se, is_transposeConv=False
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

        return blocks

    def Feature_extraction_down(self, features):
        x = features
        down_features = []
        for stage_id, block_idx in enumerate(self.fea_block_group_info_down):
            # print(stage_id)  #0
            depth = self.feature_runtime_depth_down[stage_id]
            active_idx = block_idx[:depth]
            # print(self.feature_blocks_down[:depth])
            for idx in active_idx:
                x = self.feature_blocks_down[idx](x)
            down_features.append(x)
        return down_features

    def Test_up(self):
        base_stage_width = [self.init_channel * (2 ** (i - 2)) for i in range(max(self.scale_list), 1, -1)] + [
            self.init_channel]
        stride_stages = [2] + [2 for _ in range(max(self.scale_list) - 1)]
        act_stages = ['LeakyRelu' for _ in range(max(self.scale_list))]  # relu
        se_stages = [False for _ in range(self.max_scale)]
        width_list = []
        scale_list = [i for i in range(max(self.scale_list))]
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)
        # inverted residual blocks
        blocks = []
        disp_regrs = []
        disp_ups = []
        block2s = []
        _block_index = 0
        feature_dim = int(math.pow(2, max(self.scale_list) - 1 + numpy.log2(int(self.init_channel))))  # 1024-32
        for width, s, act_func, use_se, scale in zip(width_list, stride_stages, act_stages, se_stages, scale_list):
            output_channel = width
            disp_regr = nn.Conv2d(feature_dim, 1, 3, 1, 1, bias=False)
            disp_up = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            if scale == max(self.scale_list) - 1:
                block2 = nn.ConvTranspose2d(feature_dim + 3 + 1, output_channel, 3, 1, 1)
            else:
                block2 = nn.ConvTranspose2d(feature_dim + 1, output_channel, 3, 1, 1)

            mobile_inverted_conv = DynamicMBConvLayerTest(
                in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                stride=s, act_func=act_func, use_se=use_se, is_transposeConv=True, depth_list=max(self.depth_list)
            )
            if s == 1 and feature_dim == output_channel:
                shortcut = IdentityLayer(feature_dim, feature_dim)
            else:
                shortcut = None
            blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
            feature_dim = output_channel
            disp_regrs.append(disp_regr)
            disp_ups.append(disp_up)
            block2s.append(block2)
        self.disp_regrs = nn.ModuleList(disp_regrs)
        self.disp_ups = nn.ModuleList(disp_ups)
        self.block2s = nn.ModuleList(block2s)
        return blocks

    def Feature_extraction_test_up(self, inputs, down_feas, disps):
        up_feas = []
        final_disps = []
        for i in range(len(self.feature_blocks_up)):
            if i == 0:
                bottom_fea1 = down_feas[-1]
                bottom_fea2 = down_feas[-2]
            elif 1 <= i < max(self.scale_list) - 1:
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = down_feas[max(self.scale_list) - i - 2]
            else:
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = inputs[:, :3, :, :]
            argument3 = disps[i]
            disp = self.disp_regrs[i](bottom_fea1) + argument3
            upsampled_disp = self.disp_ups[i](disp)
            bottom_fea1 = self.feature_blocks_up[i](bottom_fea1)

            concat_fea = torch.cat((bottom_fea1, upsampled_disp, bottom_fea2), 1)
            block2 = self.block2s[i](concat_fea)

            up_feas.append(block2)
            final_disps.append(disp)

        last_res_disp = self.last_disp_regr(up_feas[-1])
        final_disp = disps[-1] + last_res_disp
        final_disps.append(final_disp)
        final_disps = [self.relu(final_disp) for final_disp in final_disps]

        return final_disps

    def ConstructFeatureNet_up(self):
        base_stage_width = [self.init_channel * (2 ** (i - 2)) for i in range(max(self.scale_list), 1, -1)] + [
            self.init_channel]
        stride_stages = [2] + [2 for _ in range(max(self.scale_list) - 1)]
        act_stages = ['LeakyRelu' for _ in range(max(self.scale_list))]  # relu
        # se_stages = [False if i % 2 == 0 else True for i in range(self.max_scale)]
        se_stages = [False for _ in range(self.max_scale)]
        n_block_list = [max(self.depth_list)] * max(self.scale_list)
        width_list = []
        scale_list = [i for i in range(max(self.scale_list))]
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)
        # inverted residual blocks
        self.fea_block_group_info_up = []
        blocks = []
        disp_regrs = []
        disp_ups = []
        block2s = []
        _block_index = 0
        feature_dim = int(math.pow(2, max(self.scale_list) - 1 + numpy.log2(int(self.init_channel))))  # 1024-32
        for width, n_block, s, act_func, use_se, scale in zip(width_list, n_block_list,
                                                              stride_stages, act_stages, se_stages, scale_list):
            self.fea_block_group_info_up.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            disp_regr = nn.Conv2d(feature_dim, 1, 3, 1, 1, bias=False)
            disp_up = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            if scale == max(self.scale_list) - 1:
                block2 = nn.ConvTranspose2d(feature_dim + 3 + 1, output_channel, 3, 1, 1)
            else:
                block2 = nn.ConvTranspose2d(feature_dim + 1, output_channel, 3, 1, 1)
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                    kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                    stride=stride, act_func=act_func, use_se=use_se, is_transposeConv=True
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

            disp_regrs.append(disp_regr)
            disp_ups.append(disp_up)
            block2s.append(block2)
        self.disp_regrs = nn.ModuleList(disp_regrs)
        self.disp_ups = nn.ModuleList(disp_ups)
        self.block2s = nn.ModuleList(block2s)
        return blocks

    def Feature_extraction_up(self, inputs, down_feas, disps):

        up_feas = []
        final_disps = []

        for stage_id, block_idx in enumerate(self.fea_block_group_info_up):
            depth = self.feature_runtime_depth_up[stage_id]
            active_idx = block_idx[:depth]
            if stage_id == 0:
                bottom_fea1 = down_feas[-1]
                bottom_fea2 = down_feas[-2]
            elif 1 <= stage_id < max(self.scale_list) - 1:
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = down_feas[max(self.scale_list) - stage_id - 2]
            else:
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = inputs[:, :3, :, :]
            argument3 = disps[stage_id]
            disp = self.disp_regrs[stage_id](bottom_fea1) + argument3
            upsampled_disp = self.disp_ups[stage_id](disp)
            for idx in active_idx:
                bottom_fea1 = self.feature_blocks_up[idx](bottom_fea1)

            concat_fea = torch.cat((bottom_fea1, upsampled_disp, bottom_fea2), 1)
            block2 = self.block2s[stage_id](concat_fea)

            up_feas.append(block2)
            final_disps.append(disp)

        last_res_disp = self.last_disp_regr(up_feas[-1])
        final_disp = disps[-1] + last_res_disp
        final_disps.append(final_disp)
        final_disps = [self.relu(final_disp) for final_disp in final_disps]

        return final_disps

    def forward(self, inputs, disps):
        # down_feas = self.Feature_extraction_down(inputs)
        # down_feas = self.Feature_extraction_Test_down(inputs)
        x = inputs
        down_features = []
        for i in range(len(self.feature_blocks_down)):
            x = self.feature_blocks_down[i](x)
            down_features.append(x)

        # final_disps = self.Feature_extraction_test_up(inputs, down_feas, disps)
        # final_disps = self.Feature_extraction_up(inputs, down_feas, disps)
        up_feas = []
        final_disps = []
        for i in range(len(self.feature_blocks_up)):
            if i == 0:
                bottom_fea1 = down_features[-1]
                bottom_fea2 = down_features[-2]
            elif 1 <= i < max(self.scale_list) - 1:
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = down_features[max(self.scale_list) - i - 2]
            else:
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = inputs[:, :3, :, :]

            # block2, disp = self.feature_blocks_up[i](bottom_fea1, bottom_fea2, disps[i])

            argument3 = disps[i]
            disp = self.disp_regrs[i](bottom_fea1) + argument3
            upsampled_disp = self.disp_ups[i](disp)
            bottom_fea1 = self.feature_blocks_up[i](bottom_fea1)

            concat_fea = torch.cat((bottom_fea1, upsampled_disp, bottom_fea2), 1)
            block2 = self.block2s[i](concat_fea)

            up_feas.append(block2)
            final_disps.append(disp)

        last_res_disp = self.last_disp_regr(up_feas[-1])
        final_disp = disps[-1] + last_res_disp
        final_disps.append(final_disp)
        final_disps = [self.relu(final_disp) for final_disp in final_disps]

        return final_disps

    def set_active_subnet(self, ks_down=None, es_down=None, ds_down=None, ks_up=None, es_up=None, ds_up=None, **kwargs):

        # conv part
        net_ks_down = val2list(ks_down, len(self.feature_blocks_down))
        net_expand_ratio_down = val2list(es_down, len(self.feature_blocks_down))
        net_depth_down = val2list(ds_down, len(self.fea_block_group_info_down))

        for block, k, e in zip(self.feature_blocks_down, net_ks_down, net_expand_ratio_down):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(net_depth_down):
            if d is not None:
                self.feature_runtime_depth_down[i] = min(len(self.fea_block_group_info_down[i]), d)

        # deconv part
        net_ks_up = val2list(ks_up, len(self.feature_blocks_up))
        net_expand_ratio_up = val2list(es_up, len(self.feature_blocks_up))
        net_depth_up = val2list(ds_up, len(self.fea_block_group_info_up))

        for block, k, e in zip(self.feature_blocks_up, net_ks_up, net_expand_ratio_up):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(net_depth_up):
            if d is not None:
                self.feature_runtime_depth_up[i] = min(len(self.fea_block_group_info_up[i]), d)

    def set_active_subnet_test(self, ks_down=None, es_down=None, ds_down=None, ks_up=None, es_up=None, ds_up=None,
                               **kwargs):
        # conv part
        net_ks_down = val2list(ks_down, len(self.feature_blocks_down))
        net_expand_ratio_down = val2list(es_down, len(self.feature_blocks_down))
        net_depth_down = val2list(ds_down, len(self.feature_blocks_down))

        for block, k, e, d in zip(self.feature_blocks_down, net_ks_down, net_expand_ratio_down, net_depth_down):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e
            if d is not None:
                block.conv.active_depth = d
        # deconv part
        net_ks_up = val2list(ks_up, len(self.feature_blocks_up))
        net_expand_ratio_up = val2list(es_up, len(self.feature_blocks_up))
        net_depth_up = val2list(ds_up, len(self.feature_blocks_down))

        for block, k, e, d in zip(self.feature_blocks_up, net_ks_up, net_expand_ratio_up, net_depth_up):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e
            if d is not None:
                block.conv.active_depth = d

    def sample_active_subnet_test(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample kernel size
        ks_setting_down = []
        ks_setting_up = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.feature_blocks_down))]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting_down.append(k)
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting_up.append(k)

        # sample expand ratio
        expand_setting_down = []
        expand_setting_up = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.feature_blocks_down))]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting_down.append(e)
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting_up.append(e)

        # sample depth
        depth_setting_down = []
        depth_setting_up = []
        if not isinstance(depth_candidates[0], list):  # depth_candidates=[4]
            depth_candidates = [depth_candidates for _ in range(len(self.feature_blocks_down))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting_down.append(d)
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting_up.append(d)

        self.set_active_subnet_test(ks_setting_down, expand_setting_down, depth_setting_down, ks_setting_up,
                                    expand_setting_up, depth_setting_up)

        return {
            'refine_k_down': ks_setting_down,
            'refine_e_down': expand_setting_down,
            'refine_d_down': depth_setting_down,
            'refine_k_up': ks_setting_up,
            'refine_e_up': expand_setting_up,
            'refine_d_up': depth_setting_up
        }

    def sample_active_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample kernel size
        ks_setting_down = []
        ks_setting_up = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.feature_blocks_down))]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting_down.append(k)
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting_up.append(k)

        # sample expand ratio
        expand_setting_down = []
        expand_setting_up = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.feature_blocks_down))]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting_down.append(e)
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting_up.append(e)

        # sample depth
        depth_setting_down = []
        depth_setting_up = []
        if not isinstance(depth_candidates[0], list):  # depth_candidates=[4]
            depth_candidates = [depth_candidates for _ in
                                range(len(self.fea_block_group_info_down))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting_down.append(d)
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting_up.append(d)

        self.set_active_subnet(ks_setting_down, expand_setting_down, depth_setting_down, ks_setting_up,
                               expand_setting_up, depth_setting_up)

        return {
            'refine_k_down': ks_setting_down,
            'refine_e_down': expand_setting_down,
            'refine_d_down': depth_setting_down,
            'refine_k_up': ks_setting_up,
            'refine_e_up': expand_setting_up,
            'refine_d_up': depth_setting_up
        }

    def get_active_subnet(self, preserve_weight=True):
        # conv part
        blocks_down = []
        input_channel_down = 11
        subnet_block_info_down = []
        subnet_block_idx_down = 0
        for stage_id, block_idx in enumerate(self.fea_block_group_info_down):
            depth = self.feature_runtime_depth_down[stage_id]  # checkpoint OK
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(ResidualBlock(
                    self.feature_blocks_down[idx].conv.get_active_subnet(input_channel_down, preserve_weight),
                    copy.deepcopy(self.feature_blocks_down[idx].shortcut)
                ))
                input_channel_down = stage_blocks[-1].conv.out_channels
            blocks_down += stage_blocks
            subnet_block_info_down.append([subnet_block_idx_down + i for i in range(depth)])
            subnet_block_idx_down += depth
        _subnet_feature_blocks_down = nn.ModuleList(blocks_down)  # checkpoint OK   =4

        # deconv part
        blocks_up = []
        input_channel_up = int(math.pow(2, len(self.scale_list) - 1 + numpy.log2(int(self.init_channel))))
        subnet_block_info_up = []
        subnet_block_idx_up = 0
        for stage_id, block_idx in enumerate(self.fea_block_group_info_up):
            depth = self.feature_runtime_depth_up[stage_id]  # checkpoint OK
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(ResidualBlock(
                    self.feature_blocks_up[idx].conv.get_active_subnet(input_channel_up, preserve_weight),
                    copy.deepcopy(self.feature_blocks_up[idx].shortcut)
                ))
                input_channel_up = stage_blocks[-1].conv.out_channels
            blocks_up += stage_blocks
            subnet_block_info_up.append([subnet_block_idx_up + i for i in range(depth)])
            subnet_block_idx_up += depth
        _subnet_feature_blocks_up = nn.ModuleList(blocks_up)  # checkpoint OK   =4

        set_bn_param(_subnet_feature_blocks_down, **get_bn_param(self.feature_blocks_down))
        set_bn_param(_subnet_feature_blocks_up, **get_bn_param(self.feature_blocks_up))

        return _subnet_feature_blocks_down, subnet_block_info_down, _subnet_feature_blocks_up, subnet_block_info_up

    def get_active_subnet_test(self, preserve_weight=True):
        # conv part
        blocks_down = []
        input_channel_down = 11
        for i in range(len(self.feature_blocks_down)):
            blocks_down.append(ResidualBlock(
                self.feature_blocks_down[i].conv.get_active_subnet(input_channel_down, preserve_weight),
                copy.deepcopy(self.feature_blocks_down[i].shortcut)
            ))
            input_channel_down = blocks_down[-1].conv.out_channels
        _subnet_feature_blocks_down = nn.ModuleList(blocks_down)  # checkpoint OK   =4

        # deconv part
        blocks_up = []
        input_channel_up = int(math.pow(2, max(self.scale_list) - 1 + numpy.log2(int(self.init_channel))))
        for i in range(len(self.feature_blocks_up)):
            blocks_up.append(ResidualBlock(
                self.feature_blocks_up[i].conv.get_active_subnet(input_channel_up, preserve_weight),
                copy.deepcopy(self.feature_blocks_up[i].shortcut)
            ))
            input_channel_up = blocks_up[-1].conv.out_channels
        _subnet_feature_blocks_up = nn.ModuleList(blocks_up)  # checkpoint OK   =4

        # set_bn_param(_subnet_feature_blocks_down, **get_bn_param(self.feature_blocks_down))
        # set_bn_param(_subnet_feature_blocks_up, **get_bn_param(self.feature_blocks_up))

        return _subnet_feature_blocks_down, _subnet_feature_blocks_up

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.feature_blocks_down:
            block.conv.re_organize_middle_weights(expand_ratio_stage, is_transpose=False)
        for block in self.feature_blocks_up:
            block.conv.re_organize_middle_weights(expand_ratio_stage, is_transpose=True)


class UCNet(nn.Module):

    def __init__(self, maxdisp=192, scale=6, init_channel=32):

        super(UCNet, self).__init__()

        # self.disp_range = [maxdisp // 3 * 2 // (2**i) for i in range(scale)]
        self.disp_range = [128 + 16, 64 + 16, 32 + 8, 16 + 8, 8, 4]
        self.scale = scale
        self.init_channel = init_channel

        self.down_layers = []
        self.up_layers = []

        for i in range(scale):
            input_channel = init_channel * (2 ** (i - 1))
            output_channel = init_channel * (2 ** (i))
            corr_channel = self.disp_range[i]
            up_input_channel = output_channel
            up_output_channel = input_channel
            if i == 0:
                self.down_layers.append(
                    MatchLayer(maxdisp=corr_channel, input_channel=3, output_channel=init_channel))
            else:
                self.down_layers.append(
                    MatchLayer(maxdisp=corr_channel, input_channel=input_channel, output_channel=output_channel))

            if i == 0:
                self.up_layers.append(UpSamplingLayer(input_channel=init_channel, output_channel=init_channel,
                                                      inter_channel=init_channel + 3 + 1))
            else:
                self.up_layers.append(
                    UpSamplingLayer(input_channel=up_input_channel, output_channel=up_output_channel))

        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)

        self.last_disp_regr = nn.Conv2d(init_channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, inputs):

        imgs = torch.chunk(inputs, 2, dim=1)
        left_img = imgs[0]
        right_img = imgs[1]

        left_feas = []
        right_feas = []
        corr_feas = []
        up_feas = []
        disps = []

        for i in range(0, self.scale):
            if i == 0:
                left_fea, right_fea, corr_fea = self.down_layers[i](left_img, right_img)
            else:
                left_fea, right_fea, corr_fea = self.down_layers[i](left_feas[-1], right_feas[-1])

            left_feas.append(left_fea)
            right_feas.append(right_fea)
            corr_feas.append(corr_fea)

        for i in range(self.scale, 0, -1):
            if i == self.scale:
                up_fea, disp = self.up_layers[i - 1](corr_feas[i - 1], corr_feas[i - 2])
            elif i > 1:
                up_fea, disp = self.up_layers[i - 1](up_feas[-1], corr_feas[i - 2])
            else:
                left_img = inputs[:, :3, :, :]
                up_fea, disp = self.up_layers[i - 1](up_feas[-1], left_img)

            up_feas.append(up_fea)
            disps.append(disp)

        last_disp = self.last_disp_regr(up_feas[-1])
        last_disp = self.relu(last_disp)
        disps.append(last_disp)

        return disps


#
# class UCNetV2(nn.Module):
#     def __init__(self, init_channel=32, width_mult=1.0,
#                  ks_list=7, expand_ratio_list=8, depth_list=4, scale_list=4):
#         super(UCNetV2, self).__init__()


class OFAUCNet(nn.Module):

    def __init__(self, init_channel=32, width_mult=1.0,
                 ks_list=7, expand_ratio_list=8, depth_list=4, scale_list=6):
        super(OFAUCNet, self).__init__()
        self.init_channel = init_channel
        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.scale_list = val2list(scale_list, 1)
        self.max_scale = max(self.scale_list)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()
        self.scale_list.sort()
        self.super_disp_range = [128 + 16, 64 + 16, 32 + 8, 16 + 8, 8, 4]
        self.disp_range = [self.super_disp_range[i] for i in range(max(self.scale_list))]

        # self.feature_blocks_down = nn.ModuleList(self.ConstructFeatureNet_down())
        self.feature_blocks_down = nn.ModuleList(self.Test_down())
        # self.feature_runtime_depth_down = [len(block_idx) for block_idx in self.fea_block_group_info_down]
        # self.feature_blocks_up = nn.ModuleList(self.ConstructFeatureNet_up())
        self.feature_blocks_up = nn.ModuleList(self.Test_up())
        # self.feature_blocks_up = self.is_decoder_ofa()

        # self.feature_runtime_depth_up = [len(block_idx) for block_idx in self.fea_block_group_info_up]
        self.last_disp_regr = nn.Conv2d(init_channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.corr_act_down = nn.LeakyReLU(0.1, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def is_decoder_ofa(self):
        up_layers = []
        for i in range(max(self.scale_list)):
            input_channel = self.init_channel * (2 ** (i - 1))
            output_channel = self.init_channel * (2 ** (i))
            up_input_channel = output_channel
            up_output_channel = input_channel

            if i == 0:
                up_layers.append(UpSamplingLayer(input_channel=self.init_channel, output_channel=self.init_channel,
                                                 inter_channel=self.init_channel + 3 + 1))
            else:
                up_layers.append(
                    UpSamplingLayer(input_channel=up_input_channel, output_channel=up_output_channel))
        up_layers.reverse()
        up_layers = nn.ModuleList(up_layers)
        return up_layers

    def Test_down(self):
        base_stage_width = [self.init_channel] + [self.init_channel * (2 ** i) for i in range(1, max(self.scale_list))]
        stride_stages = [2] + [2 for _ in range(max(self.scale_list) - 1)]
        act_stages = ['relu' for _ in range(max(self.scale_list))]  # relu
        se_stages = [False for _ in range(self.max_scale)]
        disps = [i for i in range(len(self.disp_range))]
        width_list = []
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)
        fusion_blocks = []
        blocks = []
        _block_index = 0
        feature_dim = 3
        for width, s, act_func, use_se, disp in zip(width_list, stride_stages, act_stages, se_stages, disps):
            output_channel = width
            mobile_inverted_conv = DynamicMBConvLayerTest(
                in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                stride=s, act_func=act_func, use_se=use_se, is_transposeConv=False, depth_list=max(self.depth_list)
            )
            if s == 1 and feature_dim == output_channel:
                shortcut = IdentityLayer(feature_dim, feature_dim)
            else:
                shortcut = None
            blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
            feature_dim = output_channel

            fusion_block = ResBlock(output_channel + self.disp_range[disp], output_channel, stride=1)
            fusion_blocks.append(fusion_block)
        self.fusion_blocks = nn.ModuleList(fusion_blocks)
        return blocks

    def Feature_extraction_test_down(self, inputs):
        imgs = torch.chunk(inputs, 2, dim=1)
        left_img = imgs[0]
        right_img = imgs[1]

        left_feas = []
        right_feas = []
        corr_feas = []
        for i in range(len(self.feature_blocks_down)):
            if i == 0:
                left_fea = left_img
                right_fea = right_img
            else:
                left_fea = left_feas[-1]
                right_fea = right_feas[-1]
            left_fea = self.feature_blocks_down[i](left_fea)
            right_fea = self.feature_blocks_down[i](right_fea)
            corr_fea = build_corr(left_fea, right_fea, max_disp=self.disp_range[i])
            corr_fea = self.corr_act_down(corr_fea)
            concat_corr = torch.cat((left_fea, corr_fea), 1)
            concat_corr = self.fusion_blocks[i](concat_corr)

            left_feas.append(left_fea)
            right_feas.append(right_fea)
            corr_feas.append(concat_corr)

        return corr_feas

    def ConstructFeatureNet_down(self):
        base_stage_width = [self.init_channel] + [self.init_channel * (2 ** i) for i in range(1, max(self.scale_list))]
        stride_stages = [2] + [2 for _ in range(max(self.scale_list) - 1)]
        act_stages = ['relu' for _ in range(max(self.scale_list))]  # relu
        # se_stages = [False if i % 2 == 0 else True for i in range(self.max_scale)]
        se_stages = [False for _ in range(self.max_scale)]
        disps = [i for i in range(len(self.disp_range))]
        n_block_list = [max(self.depth_list)] * max(self.scale_list)
        width_list = []
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)
        # inverted residual blocks
        self.fea_block_group_info_down = []
        fusion_blocks = []
        blocks = []
        _block_index = 0
        feature_dim = 3
        for width, n_block, s, act_func, use_se, disp in zip(width_list, n_block_list,
                                                             stride_stages, act_stages, se_stages, disps):
            self.fea_block_group_info_down.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                    kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                    stride=stride, act_func=act_func, use_se=use_se, is_transposeConv=False
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
            fusion_block = ResBlock(output_channel + self.disp_range[disp], output_channel, stride=1)
            fusion_blocks.append(fusion_block)
        self.fusion_blocks = nn.ModuleList(fusion_blocks)
        return blocks

    def Feature_extraction_down(self, inputs):
        imgs = torch.chunk(inputs, 2, dim=1)
        left_img = imgs[0]
        right_img = imgs[1]

        left_feas = []
        right_feas = []
        corr_feas = []
        for stage_id, block_idx in enumerate(self.fea_block_group_info_down):
            depth = self.feature_runtime_depth_down[stage_id]
            active_idx = block_idx[:depth]
            if stage_id == 0:
                left_fea = left_img
                right_fea = right_img
            else:
                left_fea = left_feas[-1]
                right_fea = right_feas[-1]
            for idx in active_idx:
                left_fea = self.feature_blocks_down[idx](left_fea)
                right_fea = self.feature_blocks_down[idx](right_fea)
            corr_fea = build_corr(left_fea, right_fea, max_disp=self.disp_range[stage_id])
            corr_fea = self.corr_act_down(corr_fea)
            concat_corr = torch.cat((left_fea, corr_fea), 1)
            concat_corr = self.fusion_blocks[stage_id](concat_corr)

            left_feas.append(left_fea)
            right_feas.append(right_fea)
            corr_feas.append(concat_corr)

        return corr_feas

    def Test_up(self):
        base_stage_width = [self.init_channel * (2 ** (i - 2)) for i in range(max(self.scale_list), 1, -1)] + [
            self.init_channel]  # [[64, 32, 16, 8, 4, 4]]
        stride_stages = [2] + [2 for _ in range(max(self.scale_list) - 1)]
        act_stages = ['LeakyRelu' for _ in range(max(self.scale_list))]
        se_stages = [False for i in range(self.max_scale)]
        width_list = []
        scale_list = [i for i in range(max(self.scale_list))]
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)
        # inverted residual blocks
        blocks = []
        disp_regrs = []
        disp_ups = []
        block2s = []
        _block_index = 0
        feature_dim = int(math.pow(2, max(self.scale_list) - 1 + numpy.log2(int(self.init_channel))))  # 1024-32
        for width, s, act_func, use_se, scale in zip(width_list,
                                                     stride_stages, act_stages, se_stages, scale_list):
            output_channel = width
            disp_regr = nn.Conv2d(feature_dim, 1, 3, 1, 1, bias=False)
            disp_up = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            if scale == max(self.scale_list) - 1:
                block2 = nn.ConvTranspose2d(feature_dim + 3 + 1, output_channel, 3, 1, 1)
            else:
                block2 = nn.ConvTranspose2d(feature_dim + 1, output_channel, 3, 1, 1)
            mobile_inverted_conv = DynamicMBConvLayerTest(
                in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                stride=s, act_func=act_func, use_se=use_se, is_transposeConv=True, depth_list=max(self.depth_list)
            )
            if s == 1 and feature_dim == output_channel:
                shortcut = IdentityLayer(feature_dim, feature_dim)
            else:
                shortcut = None
            blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
            feature_dim = output_channel

            disp_regrs.append(disp_regr)
            disp_ups.append(disp_up)
            block2s.append(block2)
        self.disp_regrs = nn.ModuleList(disp_regrs)
        self.disp_ups = nn.ModuleList(disp_ups)
        self.block2s = nn.ModuleList(block2s)
        return blocks

    def Feature_extraction_test_up(self, inputs, corr_feas):
        up_feas = []
        disps = []
        for i in range(len(self.feature_blocks_up)):
            if i == 0:
                bottom_fea1 = corr_feas[-1]  # 5
                bottom_fea2 = corr_feas[-2]  # 4
            elif 1 <= i < (max(self.scale_list) - 1):
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = corr_feas[max(self.scale_list) - i - 2]
            else:
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = inputs[:, :3, :, :]
            disp = self.disp_regrs[i](bottom_fea1)
            upsampled_disp = self.disp_ups[i](disp)
            bottom_fea1 = self.feature_blocks_up[i](bottom_fea1)
            concat_fea = torch.cat((bottom_fea1, upsampled_disp, bottom_fea2), 1)
            block2 = self.block2s[i](concat_fea)
            up_feas.append(block2)
            disps.append(disp)

        last_disp = self.last_disp_regr(up_feas[-1])
        last_disp = self.relu(last_disp)
        disps.append(last_disp)

        return disps

    def ConstructFeatureNet_up(self):
        base_stage_width = [self.init_channel * (2 ** (i - 2)) for i in range(max(self.scale_list), 1, -1)] + [
            self.init_channel]  # [[64, 32, 16, 8, 4, 4]]
        stride_stages = [2] + [2 for _ in range(max(self.scale_list) - 1)]
        act_stages = ['LeakyRelu' for _ in range(max(self.scale_list))]
        # se_stages = [False if i % 2 == 0 else True for i in range(self.max_scale)]
        se_stages = [False for i in range(self.max_scale)]
        n_block_list = [max(self.depth_list)] * max(self.scale_list)
        width_list = []
        scale_list = [i for i in range(max(self.scale_list))]
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)
        # inverted residual blocks
        self.fea_block_group_info_up = []
        blocks = []
        disp_regrs = []
        disp_ups = []
        block2s = []
        _block_index = 0
        feature_dim = int(math.pow(2, max(self.scale_list) - 1 + numpy.log2(int(self.init_channel))))  # 1024-32
        for width, n_block, s, act_func, use_se, scale in zip(width_list, n_block_list,
                                                              stride_stages, act_stages, se_stages, scale_list):
            self.fea_block_group_info_up.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            disp_regr = nn.Conv2d(feature_dim, 1, 3, 1, 1, bias=False)
            disp_up = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
            if scale == max(self.scale_list) - 1:
                block2 = nn.ConvTranspose2d(feature_dim + 3 + 1, output_channel, 3, 1, 1)
            else:
                block2 = nn.ConvTranspose2d(feature_dim + 1, output_channel, 3, 1, 1)
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                    kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                    stride=stride, act_func=act_func, use_se=use_se, is_transposeConv=True
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

            disp_regrs.append(disp_regr)
            disp_ups.append(disp_up)
            block2s.append(block2)
        self.disp_regrs = nn.ModuleList(disp_regrs)
        self.disp_ups = nn.ModuleList(disp_ups)
        self.block2s = nn.ModuleList(block2s)
        return blocks

    def Feature_extraction_up(self, inputs, corr_feas):
        up_feas = []
        disps = []
        for stage_id, block_idx in enumerate(self.fea_block_group_info_up):
            depth = self.feature_runtime_depth_up[stage_id]
            active_idx = block_idx[:depth]
            if stage_id == 0:
                bottom_fea1 = corr_feas[-1]  # 5
                bottom_fea2 = corr_feas[-2]  # 4
            elif 1 <= stage_id < (max(self.scale_list) - 1):
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = corr_feas[max(self.scale_list) - stage_id - 2]
            else:
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = inputs[:, :3, :, :]
            disp = self.disp_regrs[stage_id](bottom_fea1)
            upsampled_disp = self.disp_ups[stage_id](disp)
            for idx in active_idx:
                bottom_fea1 = self.feature_blocks_up[idx](bottom_fea1)
            # print(bottom_fea1.shape)
            # print(upsampled_disp.shape)
            # print(bottom_fea2.shape)
            # print(stage_id)
            concat_fea = torch.cat((bottom_fea1, upsampled_disp, bottom_fea2), 1)
            # print(self.block2s)
            block2 = self.block2s[stage_id](concat_fea)
            up_feas.append(block2)
            disps.append(disp)

        last_disp = self.last_disp_regr(up_feas[-1])
        last_disp = self.relu(last_disp)
        disps.append(last_disp)

        return disps

    def forward(self, inputs):
        # corr_feas = self.Feature_extraction_down(inputs)
        # corr_feas = self.Feature_extraction_test_down(inputs)
        imgs = torch.chunk(inputs, 2, dim=1)
        left_img = imgs[0]
        right_img = imgs[1]

        left_feas = []
        right_feas = []
        corr_feas = []
        for i in range(len(self.feature_blocks_down)):
            if i == 0:
                left_fea = left_img
                right_fea = right_img
            else:
                left_fea = left_feas[-1]
                right_fea = right_feas[-1]
            left_fea = self.feature_blocks_down[i](left_fea)
            right_fea = self.feature_blocks_down[i](right_fea)
            corr_fea = build_corr(left_fea, right_fea, max_disp=self.disp_range[i])
            corr_fea = self.corr_act_down(corr_fea)
            concat_corr = torch.cat((left_fea, corr_fea), 1)
            concat_corr = self.fusion_blocks[i](concat_corr)

            left_feas.append(left_fea)
            right_feas.append(right_fea)
            corr_feas.append(concat_corr)

        # disps = self.Feature_extraction_test_up(inputs, corr_feas)
        # disps = self.Feature_extraction_up(inputs, corr_feas)
        up_feas = []
        disps = []

        for i in range(len(self.feature_blocks_up)):
            if i == 0:
                bottom_fea1 = corr_feas[-1]  # 5
                bottom_fea2 = corr_feas[-2]  # 4
            elif 1 <= i < (max(self.scale_list) - 1):
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = corr_feas[max(self.scale_list) - i - 2]
            else:
                bottom_fea1 = up_feas[-1]
                bottom_fea2 = inputs[:, :3, :, :]

            # block2, disp = self.feature_blocks_up[i](bottom_fea1, bottom_fea2)

            disp = self.disp_regrs[i](bottom_fea1)
            upsampled_disp = self.disp_ups[i](disp)
            bottom_fea1 = self.feature_blocks_up[i](bottom_fea1)
            concat_fea = torch.cat((bottom_fea1, upsampled_disp, bottom_fea2), 1)
            block2 = self.block2s[i](concat_fea)
            up_feas.append(block2)
            disps.append(disp)

        last_disp = self.last_disp_regr(up_feas[-1])
        last_disp = self.relu(last_disp)
        disps.append(last_disp)

        return disps

    def set_active_subnet(self, ks_down=None, es_down=None, ds_down=None, ks_up=None, es_up=None, ds_up=None, **kwargs):
        # conv part
        net_ks_down = val2list(ks_down, len(self.feature_blocks_down))
        net_expand_ratio_down = val2list(es_down, len(self.feature_blocks_down))
        net_depth_down = val2list(ds_down, len(self.fea_block_group_info_down))

        for block, k, e in zip(self.feature_blocks_down, net_ks_down, net_expand_ratio_down):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(net_depth_down):
            if d is not None:
                self.feature_runtime_depth_down[i] = min(len(self.fea_block_group_info_down[i]), d)

        # deconv part
        net_ks_up = val2list(ks_up, len(self.feature_blocks_up))
        net_expand_ratio_up = val2list(es_up, len(self.feature_blocks_up))
        net_depth_up = val2list(ds_up, len(self.fea_block_group_info_up))

        for block, k, e in zip(self.feature_blocks_up, net_ks_up, net_expand_ratio_up):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(net_depth_up):
            if d is not None:
                self.feature_runtime_depth_up[i] = min(len(self.fea_block_group_info_up[i]), d)

    def set_active_subnet_test(self, ks_down=None, es_down=None, ds_down=None, ks_up=None, es_up=None, ds_up=None,
                               **kwargs):
        # conv part
        net_ks_down = val2list(ks_down, len(self.feature_blocks_down))
        net_expand_ratio_down = val2list(es_down, len(self.feature_blocks_down))
        net_depth_down = val2list(ds_down, len(self.feature_blocks_down))

        for block, k, e, d in zip(self.feature_blocks_down, net_ks_down, net_expand_ratio_down, net_depth_down):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e
            if d is not None:
                block.conv.active_depth = d

        # deconv part
        net_ks_up = val2list(ks_up, len(self.feature_blocks_up))
        net_expand_ratio_up = val2list(es_up, len(self.feature_blocks_up))
        net_depth_up = val2list(ds_up, len(self.feature_blocks_up))

        for block, k, e, d in zip(self.feature_blocks_up, net_ks_up, net_expand_ratio_up, net_depth_up):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e
            if d is not None:
                block.conv.active_depth = d

    def sample_active_subnet_test(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample kernel size
        ks_setting_down = []
        ks_setting_up = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.feature_blocks_down))]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting_down.append(k)
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting_up.append(k)

        # sample expand ratio
        expand_setting_down = []
        expand_setting_up = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.feature_blocks_down))]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting_down.append(e)
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting_up.append(e)

        # sample depth
        depth_setting_down = []
        depth_setting_up = []
        if not isinstance(depth_candidates[0], list):  # depth_candidates=[4]
            depth_candidates = [depth_candidates for _ in range(len(self.feature_blocks_down))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting_down.append(d)
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting_up.append(d)

        self.set_active_subnet_test(ks_setting_down, expand_setting_down, depth_setting_down, ks_setting_up,
                                    expand_setting_up, depth_setting_up)

        return {
            'basic_k_down': ks_setting_down,
            'basic_e_down': expand_setting_down,
            'basic_d_down': depth_setting_down,
            'basic_k_up': ks_setting_up,
            'basic_e_up': expand_setting_up,
            'basic_d_up': depth_setting_up
        }

    def sample_active_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample kernel size
        ks_setting_down = []
        ks_setting_up = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.feature_blocks_down))]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting_down.append(k)
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting_up.append(k)

        # sample expand ratio
        expand_setting_down = []
        expand_setting_up = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.feature_blocks_down))]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting_down.append(e)
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting_up.append(e)

        # sample depth
        depth_setting_down = []
        depth_setting_up = []
        if not isinstance(depth_candidates[0], list):  # depth_candidates=[4]
            depth_candidates = [depth_candidates for _ in
                                range(len(self.fea_block_group_info_down))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting_down.append(d)
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting_up.append(d)

        self.set_active_subnet(ks_setting_down, expand_setting_down, depth_setting_down, ks_setting_up,
                               expand_setting_up, depth_setting_up)

        return {
            'basic_k_down': ks_setting_down,
            'basic_e_down': expand_setting_down,
            'basic_d_down': depth_setting_down,
            'basic_k_up': ks_setting_up,
            'basic_e_up': expand_setting_up,
            'basic_d_up': depth_setting_up
        }

    def get_active_subnet(self, preserve_weight=True):
        # conv part
        blocks_down = []
        input_channel_down = 3
        subnet_block_info_down = []
        subnet_block_idx_down = 0
        for stage_id, block_idx in enumerate(self.fea_block_group_info_down):
            depth = self.feature_runtime_depth_down[stage_id]  # checkpoint OK
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(ResidualBlock(
                    self.feature_blocks_down[idx].conv.get_active_subnet(input_channel_down, preserve_weight),
                    copy.deepcopy(self.feature_blocks_down[idx].shortcut)
                ))
                input_channel_down = stage_blocks[-1].conv.out_channels
            blocks_down += stage_blocks
            subnet_block_info_down.append([subnet_block_idx_down + i for i in range(depth)])
            subnet_block_idx_down += depth
        _subnet_feature_blocks_down = nn.ModuleList(blocks_down)  # checkpoint OK   =4

        # deconv part
        blocks_up = []
        input_channel_up = int(math.pow(2, len(self.scale_list) - 1 + numpy.log2(int(self.init_channel))))
        subnet_block_info_up = []
        subnet_block_idx_up = 0
        for stage_id, block_idx in enumerate(self.fea_block_group_info_up):
            depth = self.feature_runtime_depth_up[stage_id]  # checkpoint OK
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(ResidualBlock(
                    self.feature_blocks_up[idx].conv.get_active_subnet(input_channel_up, preserve_weight),
                    copy.deepcopy(self.feature_blocks_up[idx].shortcut)
                ))
                input_channel_up = stage_blocks[-1].conv.out_channels
            blocks_up += stage_blocks
            subnet_block_info_up.append([subnet_block_idx_up + i for i in range(depth)])
            subnet_block_idx_up += depth
        _subnet_feature_blocks_up = nn.ModuleList(blocks_up)  # checkpoint OK   =4

        set_bn_param(_subnet_feature_blocks_down, **get_bn_param(self.feature_blocks_down))
        set_bn_param(_subnet_feature_blocks_up, **get_bn_param(self.feature_blocks_up))

        return _subnet_feature_blocks_down, subnet_block_info_down, _subnet_feature_blocks_up, subnet_block_info_up

    def get_active_subnet_test(self, preserve_weight=True):
        # conv part
        blocks_down = []
        input_channel_down = 3
        for i in range(len(self.feature_blocks_down)):
            blocks_down.append(ResidualBlock(
                self.feature_blocks_down[i].conv.get_active_subnet_test(input_channel_down, preserve_weight),
                copy.deepcopy(self.feature_blocks_down[i].shortcut)
            ))
            input_channel_down = blocks_down[-1].conv.out_channels
        _subnet_feature_blocks_down = nn.ModuleList(blocks_down)

        # deconv part
        blocks_up = []
        input_channel_up = int(math.pow(2, max(self.scale_list) - 1 + numpy.log2(int(self.init_channel))))
        for i in range(len(self.feature_blocks_up)):
            blocks_up.append(ResidualBlock(
                self.feature_blocks_up[i].conv.get_active_subnet(input_channel_up, preserve_weight),
                copy.deepcopy(self.feature_blocks_up[i].shortcut)
            ))
            input_channel_up = blocks_up[-1].conv.out_channels
        _subnet_feature_blocks_up = nn.ModuleList(blocks_up)  # checkpoint OK   =4

        # set_bn_param(_subnet_feature_blocks_down, **get_bn_param(self.feature_blocks_down))
        # set_bn_param(_subnet_feature_blocks_up, **get_bn_param(self.feature_blocks_up))

        return _subnet_feature_blocks_down, _subnet_feature_blocks_up

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.feature_blocks_down:
            block.conv.re_organize_middle_weights(expand_ratio_stage, is_transpose=False)
        for block in self.feature_blocks_up:
            block.conv.re_organize_middle_weights(expand_ratio_stage, is_transpose=True)


class UCResNet(nn.Module):

    def __init__(self, maxdisp=192, scale=6, init_channel=32, width_mult=1.0,
                 ks_list=7, expand_ratio_list=8, depth_list=4, scale_list=6):
        super(UCResNet, self).__init__()

        self.scale = scale
        self.init_channel = init_channel

        self.is_ofa = True

        if self.is_ofa:
            self.width_mult = width_mult
            self.ks_list = val2list(ks_list, 1)
            self.expand_ratio_list = val2list(expand_ratio_list, 1)
            self.depth_list = val2list(depth_list, 1)
            self.scale_list = val2list(scale_list, 1)
            self.active_scale = max(self.scale_list)
            self.max_scale = max(self.scale_list)
            self.active_scale = self.max_scale

            self.ks_list.sort()
            self.expand_ratio_list.sort()
            self.depth_list.sort()
            self.scale_list.sort()

            self.basic_net = OFAUCNet(init_channel=self.init_channel, width_mult=self.width_mult, ks_list=self.ks_list,
                                      expand_ratio_list=self.expand_ratio_list, depth_list=self.depth_list,
                                      scale_list=self.scale_list)

            self.refine_net = OFAUCRefineNet(init_channel=self.init_channel, width_mult=self.width_mult,
                                             ks_list=self.ks_list,
                                             expand_ratio_list=self.expand_ratio_list, depth_list=self.depth_list,
                                             scale_list=self.scale_list)
        else:
            self.basic_net = UCNet(maxdisp, self.scale, self.init_channel)
            self.refine_net = UCRefineNet(self.scale, self.init_channel)

    def forward(self, inputs):
        disps = self.basic_net(inputs)
        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1 = warp_right_to_left(inputs[:, 3:, :, :], -disps[-1])
        diff_img0 = inputs[:, :3, :, :] - resampled_img1
        norm_diff_img0 = channel_length(diff_img0)
        # concat img0, img1, img1->img0, flow, diff-img
        inputs_refine = torch.cat((inputs, resampled_img1, disps[-1], norm_diff_img0), dim=1)
        refine_disps = self.refine_net(inputs_refine, disps)
        return disps, refine_disps
