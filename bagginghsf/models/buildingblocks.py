from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F

from ..models.layers import ConvCapsuleLayer3D, SwitchNorm3d
from ..models.helpers import calc_same_padding, squash


def conv3d(in_channels, out_channels, kernel_size, bias, padding, dilation=1):
    return nn.Conv3d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias,
                     dilation=dilation)


def create_conv(in_channels,
                out_channels,
                kernel_size,
                order,
                num_groups,
                padding,
                dilation=1):
    """
    Create a list of modules with together constitute a single conv layer with
    non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of
        the input
        dilation (int or tuple): Dilation factor to create dilated conv, and
        increase the recceptive field

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[
        0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(
                ('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv',
                            conv3d(in_channels,
                                   out_channels,
                                   kernel_size,
                                   bias,
                                   padding=padding,
                                   dilation=dilation)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than
            # the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm',
                            nn.GroupNorm(num_groups=num_groups,
                                         num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))

        elif char == 's':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(
                    ('SwitchNorm3d', SwitchNorm3d(in_channels, using_bn=False)))
            else:
                modules.append(
                    ('SwitchNorm3d', SwitchNorm3d(out_channels,
                                                  using_bn=False)))

        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(
                    ('InstanceNorm3d', nn.InstanceNorm3d(in_channels)))
            else:
                modules.append(
                    ('InstanceNorm3d', nn.InstanceNorm3d(out_channels)))

        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 's', 'i']"
            )

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and
    optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of
        the input
        dilation (int or tuple): Dilation factor to create dilated conv, and
        increase the recceptive field
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 order='gcr',
                 num_groups=8,
                 padding=1,
                 dilation=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels,
                                        out_channels,
                                        kernel_size,
                                        order,
                                        num_groups,
                                        padding=padding,
                                        dilation=dilation):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g.
    BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in
    order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the
    same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in
        the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of
        the input
        (first/second)_dilation (int or tuple): Dilation factor to create
        dilated conv, and increase the recceptive field
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 encoder,
                 kernel_size=3,
                 order='cr',
                 num_groups=8,
                 padding=1,
                 first_dilation=1,
                 second_dilation=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in
            # the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module(
            'SingleConv1',
            SingleConv(conv1_in_channels,
                       conv1_out_channels,
                       kernel_size,
                       order,
                       num_groups,
                       padding=padding,
                       dilation=first_dilation))
        # conv2
        self.add_module(
            'SingleConv2',
            SingleConv(conv2_in_channels,
                       conv2_out_channels,
                       kernel_size,
                       order,
                       num_groups,
                       padding=padding,
                       dilation=second_dilation))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels
    and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder
    module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity
    after the groupnorm.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 order='cr',
                 num_groups=8,
                 **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                order=order,
                                num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                order=order,
                                num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class DilatedBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by a dilated conv.
    The SingleConv takes care of increasing/decreasing the number of channels
    and also ensures that the number of output channels is compatible with
    the dilated block that follows. By using the dilated convolutions, the
    feature maps can be computed with a high spatial resolution, and the
    size of the receptive field can be enlarged arbitrarily.
    
    Motivated by: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6491864/
    `union_type` argument can be used to either `sum` or `cat` at the end of
    the dilated block.
    """

    def __init__(self,
                 in_channels,
                 out_channels_conv,
                 out_channels_dilated_conv,
                 kernel_size_conv=1,
                 kernel_size_dilated_conv=3,
                 order='cr',
                 num_groups=8,
                 dilation=2,
                 padding_conv=0,
                 padding_dilated=2,
                 union_type='cat',
                 **kwargs):
        super(DilatedBlock, self).__init__()
        self.union_type = union_type

        # first convolution
        self.conv1 = SingleConv(in_channels,
                                out_channels_conv,
                                kernel_size=kernel_size_conv,
                                order=order,
                                num_groups=num_groups,
                                dilation=1,
                                padding=padding_conv)
        # dilated block
        # remove non-linearity from the 2nd convolution since it's going to be
        # applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv2 = SingleConv(out_channels_conv,
                                out_channels_dilated_conv,
                                kernel_size=kernel_size_dilated_conv,
                                order=n_order,
                                num_groups=num_groups,
                                dilation=dilation,
                                padding=padding_dilated)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply convolutionsand save the output as a out
        out = self.conv1(x)
        out = self.conv2(out)

        # cat or sum original input
        if self.union_type == 'sum':
            out += x
        elif self.union_type == 'cat':
            out = torch.cat((out, x), dim=1)

        out = self.non_linearity(out)

        return out


class DilatedDenseNetwork(nn.Module):
    """
    A succession of dilated blocks, to construct a Dilated Dense Network.
    By using the dilated convolutions, the
    feature maps can be computed with a high spatial resolution, and the
    size of the receptive field can be enlarged arbitrarily.
    
    Motivated by: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6491864/
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of
        the input
    """

    def __init__(self,
                 in_channels,
                 out_channels_conv,
                 out_channels_dilated_conv,
                 n_blocks=6,
                 kernel_size_conv=1,
                 kernel_size_dilated_conv=3,
                 conv_layer_order='cbr',
                 num_groups=8,
                 padding_conv=0,
                 padding_dilated=2,
                 union_type='cat'):
        super(DilatedDenseNetwork, self).__init__()

        dilations = [2**i for i in range(-(-n_blocks // 2)) for _ in (0, 1)
                    ][:n_blocks]
        blocks = []
        for i, dilation in enumerate(dilations):
            # ! padding_dilated = dilation is a special case where the
            # ! kernel of the dilated conv is 3x3x3
            _in_channels = in_channels + i * out_channels_dilated_conv
            block = DilatedBlock(
                in_channels=_in_channels,
                out_channels_conv=out_channels_conv,
                out_channels_dilated_conv=out_channels_dilated_conv,
                kernel_size_conv=kernel_size_conv,
                kernel_size_dilated_conv=kernel_size_dilated_conv,
                order=conv_layer_order,
                num_groups=num_groups,
                dilation=dilation,
                union_type=union_type,
                padding_conv=padding_conv,
                padding_dilated=dilation)

            blocks.append(block)

        blocks.append(
            SingleConv(_in_channels + out_channels_dilated_conv,
                       in_channels,
                       1,
                       conv_layer_order,
                       num_groups,
                       padding=0,
                       dilation=1))

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.network(x)

        return x


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed
    by a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of
        the input
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_kernel_size=3,
                 apply_pooling=True,
                 pool_kernel_size=2,
                 pool_type='max',
                 basic_module=DoubleConv,
                 conv_layer_order='gcr',
                 num_groups=8,
                 padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels,
                                         out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation)
    followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must
            reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of
        the input
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 out_channels_dilated_conv=32,
                 conv_kernel_size=3,
                 dilated_conv_kernel_size=3,
                 scale_factor=(2, 2, 2),
                 basic_module=DoubleConv,
                 conv_layer_order='gcr',
                 num_groups=8,
                 mode='nearest',
                 padding=1,
                 use_dunet=False,
                 dunet_conv_layer_order='cr',
                 dunet_n_blocks=6,
                 dunet_num_groups=8,
                 use_attention=False,
                 normalization="s",
                 using_bn=False):
        super(Decoder, self).__init__()
        self.use_dunet = use_dunet
        self.use_attention = use_attention

        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use interpolation for
            # upsampling and concatenation joining
            self.upsampling = Upsampling(transposed_conv=False,
                                         in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=conv_kernel_size,
                                         scale_factor=scale_factor,
                                         mode=mode)
            # concat joining
            self.joining = partial(self._joining, concat=True)
        else:
            # if basic_module=ExtResNetBlock use transposed convolution
            # upsampling and summation joining
            self.upsampling = Upsampling(transposed_conv=True,
                                         in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=conv_kernel_size,
                                         scale_factor=scale_factor,
                                         mode=mode)
            # sum joining
            self.joining = partial(self._joining, concat=False)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        self.basic_module = basic_module(in_channels,
                                         out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

        if use_dunet:
            self.dilated_dense_network = DilatedDenseNetwork(
                in_channels=out_channels,
                out_channels_conv=out_channels,
                out_channels_dilated_conv=out_channels_dilated_conv,
                conv_layer_order=dunet_conv_layer_order,
                n_blocks=dunet_n_blocks,
                num_groups=dunet_num_groups)

        if use_attention:
            self.attention = AttentionConvBlock(F_g=out_channels,
                                                F_l=out_channels,
                                                F_int=out_channels // 2,
                                                F_out=1,
                                                normalization=normalization,
                                                using_bn=using_bn)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)

        if self.use_dunet:
            encoder_features = self.dilated_dense_network(encoder_features)

        if self.use_attention:
            encoder_features = self.attention(g=x, x=encoder_features)

        x = self.joining(encoder_features, x)

        x = self.basic_module(x)

        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class Upsampling(nn.Module):
    """
    Upsamples a given multi-channel 3D data using either interpolation or
    learned transposed convolution.

    Args:
        transposed_conv (bool): if True uses ConvTranspose3d for upsampling,
        otherwise uses interpolation
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'.
            Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self,
                 transposed_conv,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=3,
                 scale_factor=(2, 2, 2),
                 mode='nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            # make sure that the output size reverses the MaxPool3d from the
            # corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1)
        else:
            self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class SingleCaps(nn.Sequential):
    """
    Basic capsule module
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 strides=1,
                 routings=3,
                 activation=squash,
                 sigmoid_routing=True,
                 transposed=False,
                 constrained=True,
                 final_squash=True,
                 use_switchnorm=False):
        super(SingleCaps, self).__init__()
        o = 64
        k = kernel_size - 1 if transposed else kernel_size
        p, _ = calc_same_padding(input_=o,
                                 kernel=k,
                                 stride=strides,
                                 transposed=transposed)
        self.add_module(
            "Capsule",
            ConvCapsuleLayer3D(kernel_size=k,
                               input_num_capsule=input_num_capsule,
                               input_num_atoms=input_num_atoms,
                               num_capsule=num_capsule,
                               num_atoms=num_atoms,
                               strides=strides,
                               padding=p,
                               routings=routings,
                               sigmoid_routing=sigmoid_routing,
                               transposed=transposed,
                               constrained=constrained,
                               activation=activation,
                               final_squash=final_squash,
                               use_switchnorm=use_switchnorm))


class DoubleCaps(nn.Sequential):
    """
    A module consisting of two consecutive capsule layers
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 strides=1,
                 routings=3,
                 activation=squash,
                 sigmoid_routing=True,
                 transposed=False,
                 constrained=True,
                 final_squash=True,
                 use_switchnorm=False):
        super(DoubleCaps, self).__init__()

        if type(strides) != list:
            strides = [strides, strides]
        if type(transposed) != list:
            transposed = [transposed, transposed]
        if type(routings) != list:
            routings = [routings, routings]
        if type(activation) != list:
            activation = [activation, activation]
        if type(final_squash) != list:
            final_squash = [final_squash, final_squash]
        if type(use_switchnorm) != list:
            use_switchnorm = [use_switchnorm, use_switchnorm]
        input_capsules = [input_num_capsule, num_capsule]
        input_atoms = [input_num_atoms, num_atoms]

        for i, stride in enumerate(strides):
            o = 64
            k = kernel_size - 1 if transposed[i] else kernel_size
            p, _ = calc_same_padding(input_=o,
                                     kernel=k,
                                     stride=stride,
                                     transposed=transposed[i])
            self.add_module(
                f'Capsule{i}',
                ConvCapsuleLayer3D(kernel_size=k,
                                   input_num_capsule=input_capsules[i],
                                   input_num_atoms=input_atoms[i],
                                   num_capsule=num_capsule,
                                   num_atoms=num_atoms,
                                   strides=stride,
                                   padding=p,
                                   routings=routings[i],
                                   sigmoid_routing=sigmoid_routing,
                                   transposed=transposed[i],
                                   constrained=constrained,
                                   activation=activation[i],
                                   final_squash=final_squash[i],
                                   use_switchnorm=use_switchnorm[i]))


class EncoderCaps(nn.Module):
    """
    A single module from the encoder path consisting of a double capsule block.
    The first capsule may have a stride of 2, and the second a stride of 1, so
    that spatial dim is divided by 2.
    Args:
        todo.
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 basic_module=DoubleCaps,
                 strides=[2, 1],
                 routings=1,
                 sigmoid_routing=True,
                 transposed=False,
                 constrained=True,
                 use_switchnorm=False):
        super(EncoderCaps, self).__init__()

        self.basic_module = basic_module(kernel_size=kernel_size,
                                         input_num_capsule=input_num_capsule,
                                         input_num_atoms=input_num_atoms,
                                         num_capsule=num_capsule,
                                         num_atoms=num_atoms,
                                         strides=strides,
                                         routings=routings,
                                         sigmoid_routing=sigmoid_routing,
                                         transposed=transposed,
                                         constrained=constrained,
                                         use_switchnorm=use_switchnorm)

    def forward(self, x):
        x = self.basic_module(x)
        return x


class DecoderCaps(nn.Module):
    """
    A single module for decoder path consisting of the upsampling caps
    followed by a basic capsule.
    Args:
        todo.
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 strides=[2, 1],
                 routings=1,
                 sigmoid_routing=True,
                 transposed=[True, False],
                 constrained=True,
                 use_switchnorm=False,
                 union_type="cat"):
        super(DecoderCaps, self).__init__()
        self.union_type = union_type
        if isinstance(routings, int):
            routings = [routings] * 2

        self.transposed_caps = SingleCaps(kernel_size=kernel_size,
                                          input_num_capsule=input_num_capsule,
                                          input_num_atoms=input_num_atoms,
                                          num_capsule=num_capsule,
                                          num_atoms=num_atoms,
                                          strides=strides[0],
                                          routings=routings[0],
                                          sigmoid_routing=sigmoid_routing,
                                          transposed=transposed[0],
                                          constrained=constrained,
                                          use_switchnorm=use_switchnorm)
        _f = 2 if union_type == "cat" else 1
        self.caps = SingleCaps(kernel_size=kernel_size,
                               input_num_capsule=num_capsule * _f,
                               input_num_atoms=num_atoms,
                               num_capsule=num_capsule,
                               num_atoms=num_atoms,
                               strides=strides[1],
                               routings=routings[1],
                               sigmoid_routing=sigmoid_routing,
                               transposed=transposed[1],
                               constrained=constrained,
                               use_switchnorm=use_switchnorm)

    def forward(self, encoder_features, x):
        up = self.transposed_caps(x)
        if self.union_type == "cat":
            x = torch.cat((encoder_features, up), dim=1)
        elif self.union_type == "sum":
            x = up + encoder_features
            # todo: squash?
        x = self.caps(x)

        return x


class AttentionConvBlock(nn.Module):
    """
    3D Conv Attention Block w/ optional Normalization.
    For normalization, it supports:
    - `b` for `BatchNorm3d`,
    - `s` for `SwitchNorm3d`.
    
    `using_bn` controls SwitchNorm's behavior. It has no effect is
    `normalization == "b"`.

    SwitchNorm3d comes from:
    <https://github.com/switchablenorms/Switchable-Normalization>
    """

    def __init__(self,
                 F_g,
                 F_l,
                 F_int,
                 F_out=1,
                 normalization=None,
                 using_bn=False):
        super(AttentionConvBlock, self).__init__()

        W_g = [
            nn.Conv3d(
                F_g,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        W_x = [
            nn.Conv3d(
                F_l,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        psi = [
            nn.Conv3d(
                F_int,
                F_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        if normalization == "b":
            W_g.append(nn.BatchNorm3d(F_int))
            W_x.append(nn.BatchNorm3d(F_int))
            psi.append(nn.BatchNorm3d(F_out))
        elif normalization == "s":
            W_g.append(SwitchNorm3d(F_int, using_bn=using_bn))
            W_x.append(SwitchNorm3d(F_int, using_bn=using_bn))
            psi.append(SwitchNorm3d(F_out, using_bn=using_bn))

        self.W_g = nn.Sequential(*W_g)
        self.W_x = nn.Sequential(*W_x)

        psi.append(nn.Sigmoid())
        self.psi = nn.Sequential(*psi)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionBlock(nn.Module):
    """
    3D Caps Attention Block w/ optional Normalization.
    For normalization, it supports:
    - `b` for `BatchNorm3d`,
    - `s` for `SwitchNorm3d`.
    
    `using_bn` controls SwitchNorm's behavior. It has no effect is
    `normalization == "b"`.

    SwitchNorm3d comes from:
    <https://github.com/switchablenorms/Switchable-Normalization>
    """

    def __init__(self,
                 F_g,
                 F_l,
                 F_int,
                 F_out=1,
                 normalization=None,
                 using_bn=False):
        super(AttentionBlock, self).__init__()

        W_g = [
            nn.Conv3d(
                F_g,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        W_x = [
            nn.Conv3d(
                F_l,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        psi = [
            nn.Conv3d(
                F_int,
                F_out,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        ]
        if normalization == "b":
            W_g.append(nn.BatchNorm3d(F_int))
            W_x.append(nn.BatchNorm3d(F_int))
            psi.append(nn.BatchNorm3d(F_out))
        elif normalization == "s":
            W_g.append(SwitchNorm3d(F_int, using_bn=using_bn))
            W_x.append(SwitchNorm3d(F_int, using_bn=using_bn))
            psi.append(SwitchNorm3d(F_out, using_bn=using_bn))

        self.W_g = nn.Sequential(*W_g)
        self.W_x = nn.Sequential(*W_x)

        psi.append(nn.Sigmoid())
        self.psi = nn.Sequential(*psi)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # Reshaping
        # g & x should normally have the same shape here
        # I don't think we should be more specific right now.
        bs, C, A, a, b, c = g.shape

        g1 = self.W_g(g.view(bs, C * A, a, b, c))
        x1 = self.W_x(x.view(bs, C * A, a, b, c))
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Unsqueeze to match capsule dimension
        psi = psi.unsqueeze(1)
        out = x * psi

        return out


class AttentionDecoderCaps(nn.Module):
    """
    A single module for decoder path consisting of the upsampling caps
    followed by a basic capsule. Using encoder_features as attention gate
    Args:
        todo.
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 strides=[2, 1],
                 routings=1,
                 sigmoid_routing=True,
                 transposed=[True, False],
                 constrained=True,
                 union_type=None,
                 normalization="s",
                 using_bn=False):
        super(AttentionDecoderCaps, self).__init__()
        self.union_type = union_type
        if isinstance(routings, int):
            routings = [routings] * 2

        self.transposed_caps = SingleCaps(kernel_size=kernel_size,
                                          input_num_capsule=input_num_capsule,
                                          input_num_atoms=input_num_atoms,
                                          num_capsule=num_capsule,
                                          num_atoms=num_atoms,
                                          strides=strides[0],
                                          routings=routings[0],
                                          sigmoid_routing=sigmoid_routing,
                                          transposed=transposed[0],
                                          constrained=constrained)
        _f = 2 if union_type == "cat" else 1
        self.caps = SingleCaps(kernel_size=kernel_size,
                               input_num_capsule=num_capsule * _f,
                               input_num_atoms=num_atoms,
                               num_capsule=num_capsule,
                               num_atoms=num_atoms,
                               strides=strides[1],
                               routings=routings[1],
                               sigmoid_routing=sigmoid_routing,
                               transposed=transposed[1],
                               constrained=constrained)

        self.att = AttentionBlock(F_g=num_capsule * num_atoms,
                                  F_l=num_capsule * num_atoms,
                                  F_int=(num_capsule * num_atoms) // 2,
                                  F_out=num_atoms,
                                  normalization=normalization,
                                  using_bn=using_bn)

    def forward(self, encoder_features, x):
        up = self.transposed_caps(x)

        att = self.att(encoder_features, up)

        if self.union_type == "cat":
            x = torch.cat((att, up), dim=1)
        elif self.union_type == "sum":
            x = up + att
            # todo: squash?
        else:
            x = att
        x = self.caps(x)

        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim,
                                    out_channels=in_dim // 2,
                                    kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim,
                                  out_channels=in_dim // 2,
                                  kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim,
                                    out_channels=in_dim,
                                    kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        """
            todo;
        """
        bs, C, A, h, w, d = x.size()

        proj_query = self.query_conv(x.view(bs, C * A, h, w, d)).view(
            bs, -1, h * w * d).permute(0, 2, 1)  # B x C x (N)
        proj_key = self.key_conv(g.view(bs, C * A, h, w,
                                        d)).view(bs, -1,
                                                 h * w * d)  # B x C x (*D*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(g.view(bs, C * A, h, w,
                                            d)).view(bs, -1,
                                                     h * w * d)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, A, h, w, d)

        out = self.gamma * out + g
        return out


class ReshapeStem(nn.Module):

    def __init__(self, dim=1):
        super(ReshapeStem, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class StemCaps(nn.Sequential):

    def __init__(self,
                 input_size=64,
                 stem_size=1,
                 in_channels=1,
                 stem_channels=128,
                 stem_kernel_size=5,
                 stem_order="cr",
                 stem_dilation=1,
                 reshape=True):
        super(StemCaps, self).__init__()
        assert stem_size > 0

        if type(stem_channels) != list:
            stem_channels = [stem_channels] * stem_size
        if type(stem_kernel_size) != list:
            stem_kernel_size = [stem_kernel_size] * stem_size
        if type(stem_order) != list:
            stem_order = [stem_order] * stem_size
        if type(stem_dilation) != list:
            stem_dilation = [stem_dilation] * stem_size

        in_channels = [in_channels] + stem_channels[:-1]

        # Safety checks
        should_match_stem_size = [
            in_channels, stem_channels, stem_kernel_size, stem_order,
            stem_dilation
        ]
        for var in should_match_stem_size:
            assert len(
                var
            ) == stem_size, f"Incorrect list: {var}. Its length should match `stem_size`."

        # Adding a succession of convolutions with `same` padding
        for stem_conv in range(stem_size):
            p, _ = calc_same_padding(
                input_size,
                kernel=stem_kernel_size[stem_conv],
                dilation=stem_dilation[stem_conv],
            )
            self.add_module(
                f"StemConv{stem_conv}",
                SingleConv(in_channels[stem_conv],
                           stem_channels[stem_conv],
                           kernel_size=stem_kernel_size[stem_conv],
                           order=stem_order[stem_conv],
                           num_groups=8,
                           padding=p,
                           dilation=stem_dilation[stem_conv]))

        if reshape:
            self.add_module("ReshapeStem", ReshapeStem())


class ClassicStem(nn.Module):

    def __init__(self,
                 num_capsule=4,
                 num_atoms=32,
                 input_size=64,
                 stem_size=1,
                 in_channels=1,
                 stem_channels=128,
                 stem_kernel_size=5,
                 stem_order="cr",
                 stem_dilation=1,
                 reshape=True,
                 constrained=True,
                 use_switchnorm=False):
        super(ClassicStem, self).__init__()

        if type(stem_channels) != list:
            stem_channels = [stem_channels] * stem_size

        self.conv = StemCaps(in_channels=in_channels,
                             input_size=input_size,
                             stem_size=stem_size,
                             stem_channels=stem_channels,
                             stem_kernel_size=stem_kernel_size,
                             stem_order=stem_order,
                             stem_dilation=stem_dilation,
                             reshape=reshape)

        self.caps = SingleCaps(kernel_size=3,
                               input_num_capsule=1,
                               input_num_atoms=stem_channels[-1],
                               num_capsule=num_capsule,
                               num_atoms=num_atoms,
                               strides=1,
                               routings=1,
                               sigmoid_routing=True,
                               constrained=constrained,
                               use_switchnorm=use_switchnorm)

    def forward(self, x):
        x = self.conv(x)
        return self.caps(x)


class StemBlock(nn.Module):

    def __init__(self,
                 input_size=64,
                 n_caps=4,
                 stem_size=3,
                 in_channels=1,
                 stem_channels=[16, 16, 16],
                 stem_kernel_sizes=[[1, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
                 stem_order="cr",
                 stem_dilations=[[1, 1, 1], [6, 1, 1], [12, 1, 1], [18, 1, 1]]):
        super(StemBlock, self).__init__()

        stems = [
            StemCaps(input_size=input_size,
                     stem_size=stem_size,
                     in_channels=in_channels,
                     stem_channels=stem_channels,
                     stem_kernel_size=stem_kernel_sizes[c],
                     stem_order=stem_order,
                     stem_dilation=stem_dilations[c],
                     reshape=False) for c in range(n_caps)
        ]
        self.stems = nn.ModuleList(stems)

    def forward(self, x):
        x = torch.stack([stem(x) for stem in self.stems], dim=1)
        return x


class ResCapsBlock(nn.Module):
    """
    Basic Residual Caps block consisting of a SingleCaps followed by the residual block.
    The SingleCaps takes care of increasing/decreasing the number of channels
    and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleCaps.
    """

    def __init__(self,
                 kernel_size,
                 input_num_capsule,
                 input_num_atoms,
                 num_capsule,
                 num_atoms,
                 block_size=3,
                 strides=1,
                 routings=3,
                 sigmoid_routing=True,
                 transposed=False,
                 constrained=True,
                 skip_capsule=False,
                 skip_kernel_size=1,
                 skip_strides=1,
                 skip_num_capsule=4,
                 skip_routings=1,
                 skip_sigmoid_routing=True,
                 skip_constrained=True,
                 union_type="cat",
                 squash_sum_union=False,
                 input_size=64):
        super(ResCapsBlock, self).__init__()
        assert block_size > 2
        self.union_type = union_type
        self.skip_capsule = skip_capsule
        self.squash_sum_union = squash_sum_union

        if type(num_capsule) != list:
            num_capsule = [num_capsule] * block_size
        if type(num_atoms) != list:
            num_atoms = [num_atoms] * block_size
        if type(kernel_size) != list:
            kernel_size = [kernel_size] * block_size
        if type(strides) != list:
            strides = [strides] * block_size
        if type(routings) != list:
            routings = [routings] * block_size
        if type(sigmoid_routing) != list:
            sigmoid_routing = [sigmoid_routing] * block_size
        if type(transposed) != list:
            transposed = [transposed] * block_size
        if type(constrained) != list:
            constrained = [constrained] * block_size

        input_num_capsule = [input_num_capsule] + num_capsule[:-1]
        input_num_atoms = [input_num_atoms] + num_atoms[:-1]

        # Safety checks
        should_match_block_size = [
            input_num_capsule, input_num_atoms, num_capsule, num_atoms,
            kernel_size, strides, routings, sigmoid_routing, transposed,
            constrained
        ]
        for var in should_match_block_size:
            assert len(
                var
            ) == block_size, f"Incorrect list: {var}. Its length should match `block_size`."

        # first capsule
        p, _ = calc_same_padding(input_size, kernel=kernel_size[0])
        self.primary_caps = SingleCaps(kernel_size=kernel_size[0],
                                       input_num_capsule=input_num_capsule[0],
                                       input_num_atoms=input_num_atoms[0],
                                       num_capsule=num_capsule[0],
                                       num_atoms=num_atoms[0],
                                       strides=strides[0],
                                       padding=p,
                                       routings=routings[0],
                                       sigmoid_routing=sigmoid_routing[0],
                                       transposed=transposed[0],
                                       constrained=constrained[0])

        # subsequent capsules
        modules = []
        for i in range(1, block_size):
            p, _ = calc_same_padding(input_size, kernel=kernel_size[i])
            modules.append(
                SingleCaps(kernel_size=kernel_size[i],
                           input_num_capsule=input_num_capsule[i],
                           input_num_atoms=input_num_atoms[i],
                           num_capsule=num_capsule[i],
                           num_atoms=num_atoms[i],
                           strides=strides[i],
                           padding=p,
                           routings=routings[i],
                           sigmoid_routing=sigmoid_routing[i],
                           transposed=transposed[i],
                           constrained=constrained[i]))
        self.secondary_capsules = nn.Sequential(*modules)

        # Add a capsule in the skip connection
        if skip_capsule:
            p, _ = calc_same_padding(input_size,
                                     kernel=skip_kernel_size,
                                     stride=skip_strides)
            self.skip_cap = SingleCaps(kernel_size=skip_kernel_size,
                                       input_num_capsule=num_capsule[0],
                                       input_num_atoms=num_atoms[0],
                                       num_capsule=skip_num_capsule,
                                       num_atoms=num_atoms[-1],
                                       strides=skip_strides,
                                       padding=p,
                                       routings=skip_routings,
                                       sigmoid_routing=skip_sigmoid_routing,
                                       transposed=False,
                                       constrained=skip_constrained)

    def forward(self, x):
        # apply first capsule and save the output as a residual
        x = self.primary_caps(x)
        residual = x

        # residual block
        x = self.secondary_capsules(x)
        if self.skip_capsule:
            residual = self.skip_cap(residual)

        if self.union_type == "sum":
            x += residual
            # do we need to squash the vectors here?
            if self.squash_sum_union:
                x = squash(x, dim=2)
        elif self.union_type == "cat":
            x = torch.cat((x, residual), dim=1)

        return x
