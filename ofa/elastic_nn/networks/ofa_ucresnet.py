from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from torchinfo import summary

# from torch2trt import torch2trt
sys.path.append("home/jingbo/ucresnet/networks")
from networks.ucnet import UCResNet, OFAUCNet, OFAUCRefineNet
from networks.submodules import *
from utils.common import val2list, make_divisible
from ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer
from ofa.utils.layers import IdentityLayer, ResidualBlock
from ofa.utils.my_modules import MyNetwork, get_bn_param, set_bn_param
import copy


# from torch2trt import torch2trt


class OFAUCResNet(UCResNet):
    def __init__(self, width_mult=1.0, init_channel=32,
                 ks_list=7, expand_ratio_list=8, depth_list=4, scale_list=6):
        self.width_mult = width_mult
        self.init_channel = init_channel
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

        super(OFAUCResNet, self).__init__(init_channel=self.init_channel, width_mult=self.width_mult,
                                          ks_list=self.ks_list,
                                          expand_ratio_list=self.expand_ratio_list, depth_list=self.depth_list,
                                          scale_list=self.scale_list)

    def set_active_subnet(self, ks_down=None, es_down=None, ds_down=None, ks_up=None, es_up=None, ds_up=None, **kwargs):
        # self.basic_net.set_active_subnet(ks_down=ks_down, es_down=es_down, ds_down=ds_down, ks_up=ks_up, es_up=es_up,
        #                                  ds_up=ds_up)
        #
        # self.refine_net.set_active_subnet(ks_down=ks_down, es_down=es_down, ds_down=ds_down, ks_up=ks_up, es_up=es_up,
        #                                   ds_up=ds_up)
        self.basic_net.set_active_subnet_test(ks_down=ks_down, es_down=es_down, ds_down=ds_down, ks_up=ks_up,
                                              es_up=es_up,
                                              ds_up=ds_up)

        self.refine_net.set_active_subnet_test(ks_down=ks_down, es_down=es_down, ds_down=ds_down, ks_up=ks_up,
                                               es_up=es_up,
                                               ds_up=ds_up)

    def sample_active_subnet(self):
        settings = {}
        # basic_net_settings = self.basic_net.sample_active_subnet()
        basic_net_settings = self.basic_net.sample_active_subnet_test()
        # refine_net_settings = self.refine_net.sample_active_subnet()
        refine_net_settings = self.refine_net.sample_active_subnet_test()
        settings.update(basic_net_settings)
        settings.update(refine_net_settings)

        return settings

    def get_active_subnet(self, preserve_weight=True):
        b_down, b_up = self.basic_net.get_active_subnet_test(preserve_weight)
        r_down, r_up = self.refine_net.get_active_subnet_test(preserve_weight)

        _net = UCResNet(init_channel=self.init_channel, scale_list=self.scale_list)
        _net.basic_net = copy.deepcopy(self.basic_net)
        _net.refine_net = copy.deepcopy(self.refine_net)

        _net.basic_net.feature_blocks_down = b_down
        _net.basic_net.feature_blocks_up = b_up
        _net.refine_net.feature_blocks_down = r_down
        _net.refine_net.feature_blocks_up = r_up
        set_bn_param(_net, **get_bn_param(self))
        return _net

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if '.mobile_inverted_conv.' in key:
                new_key = key.replace('.mobile_inverted_conv.', '.conv.')
            else:
                new_key = key
            if new_key in model_dict:
                pass
            elif '.bn.bn.' in new_key:
                new_key = new_key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in new_key:
                new_key = new_key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in new_key:
                new_key = new_key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in new_key:
                new_key = new_key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in new_key:
                new_key = new_key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in new_key:
                new_key = new_key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAUCResNet, self).load_state_dict(model_dict)

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        self.basic_net.re_organize_middle_weights(expand_ratio_stage)
        self.refine_net.re_organize_middle_weights(expand_ratio_stage)


if __name__ == '__main__':
    # net=OFAUCNet(init_channel=8,scale_list=4)
    net = OFAUCRefineNet(init_channel=8, scale_list=4)
    print(net)
