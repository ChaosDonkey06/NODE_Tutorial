import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url
from temporal_shift import *
from non_local_block_ChaosDonkey06 import *


__all__ = ['resnet_tsn']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Block architecture is independet of the temporal summary (NLC/TSM)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Block architecture is independet of the temporal summary (NLC/TSM)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class TSNNet(nn.Module):
    def __init__(self, block, layers, num_classes, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(TSNNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Encoder
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        if True:
            n_segment = 2

            n_segment_list = [n_segment] * 4
            place = 'block'
            n_div=8
            

            self.layer2 = nn.Sequential(
                NL2DWrapper(self.layer2[0], n_segment),
                self.layer2[1]
            )

            self.layer3 = nn.Sequential(
                NL2DWrapper(self.layer3[0], n_segment),
                self.layer3[1]
            )
            if place == 'block':
                def make_block_temporal(stage, this_segment):
                    blocks = list(stage.children())
                    print('=> Processing stage with {} blocks'.format(len(blocks)))
                    for i, b in enumerate(blocks):
                        blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
                    return nn.Sequential(*(blocks))

                self.layer1 = make_block_temporal(self.layer1, n_segment_list[0])
                self.layer2 = make_block_temporal(self.layer2, n_segment_list[1])
                self.layer3 = make_block_temporal(self.layer3, n_segment_list[2])
                self.layer4 = make_block_temporal(self.layer4, n_segment_list[3])

        #The dropout is bets at  the final layer for eveery segment
        self.drop_layer = nn.Dropout2d(p=0.8)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_out = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # You probaby dont want to modify this
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # Utility method to perform the forward pass of a single segment
    def forward_segment(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)


        x = self.layer2(x)
        x = self.layer3(x)
        x = self.drop_layer(self.layer4(x))

        # You might want to return intermediante states here for the TSM
        return x

    # Actual forward and segmental agregation
    #def forward(self, s1, s2,):
    #    s1 = self.forward_segment(s1)
    #    s2 = self.forward_segment(s2)

        # Segmental agregation is just a stack and pool operation
    #    s = torch.stack((s1, s2))

    #    s = s.permute(1, 2, 0, 3, 4) # Time,Channel,Batch,W,H
    #    s = self.avgpool(s)

        # This is standard in a resnet
    #    s = torch.flatten(s, 1)
    #    return self.fc_out(s)

    def forward(self, x):
        n_segment=2

        #base_out = base_out.view( (-1, self.num_segments // 2) + base_out.size()[1:] )


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)


        x = self.layer2(x)
        x = self.layer3(x)
        x = self.drop_layer(self.layer4(x))
        # x.size = nt, c, w, h

        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1,2)  
        # n, c, t, h, w
                
        s=x

        s = self.avgpool(s)
        # This is standard in a resnet
        s = torch.flatten(s, 1)
        out=self.fc_out(s)
        return out


def _resnet_tsn(arch, block, layers, pretrained, progress, num_classes, **kwargs):
    model = TSNNet(block, layers, num_classes, **kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


######################
### ACTUAL NETORKS ###
######################
#Sure you can get a deeper resnet here but that is not the homework
def resnet_tsn(pretrained=True, progress=True, **kwargs):

    return _resnet_tsn('resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                       progress, num_classes=20, **kwargs)

def resnet_tsm(arch,pretrained=True,progress=True):
    import torchvision
    if arch == 'resnet18':
        model = torchvision.models.resnet18()

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)

    make_temporal_shift( model , n_segment = 2,n_div= 8, place='blockres',temporal_pool=False)
    return model