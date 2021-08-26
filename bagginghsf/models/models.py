# import numpy as np
# import pymia.evaluation.evaluator as eval_
# import pymia.evaluation.metric as metric
# import pymia.evaluation.writer as writer
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..models.losses import FocalTversky_loss, forgiving_loss
from ..models.buildingblocks import (Decoder, EncoderCaps, DecoderCaps,
                                     AttentionDecoderCaps, SingleCaps,
                                     DoubleCaps, DoubleConv, Encoder,
                                     ExtResNetBlock, ResCapsBlock, StemBlock,
                                     ClassicStem)
from ..models.helpers import (calc_same_padding, number_of_features_per_level,
                              squash, smsquash)
from ..models.layers import Length


class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        dunet_mode (string): supports 'last' (only one DUnet at the bottom of UNet), 'always', or 'first'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid,
                 basic_module,
                 f_maps=64,
                 layer_order='gcr',
                 num_groups=8,
                 num_levels=4,
                 is_segmentation=True,
                 testing=True,
                 conv_kernel_size=3,
                 pool_kernel_size=2,
                 conv_padding=1,
                 dunet=False,
                 dunet_conv_layer_order='cr',
                 dunet_n_blocks=6,
                 dunet_num_groups=8,
                 dilated_conv_kernel_size=3,
                 out_channels_dilated_conv=32,
                 dunet_mode='last',
                 **kwargs):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(
                    in_channels,
                    out_feature_num,
                    apply_pooling=False,  # skip pooling in the firs encoder
                    basic_module=basic_module,
                    conv_layer_order=layer_order,
                    conv_kernel_size=conv_kernel_size,
                    num_groups=num_groups,
                    padding=conv_padding)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                encoder = Encoder(f_maps[i - 1],
                                  out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct
            # striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)

            if dunet:
                if type(dunet_mode) == int:
                    use_dunet = True if i == dunet_mode else False
                    if isinstance(dunet_n_blocks, list):
                        n_blocks = dunet_n_blocks[dunet_mode]
                    else:
                        n_blocks = dunet_n_blocks
                if dunet_mode == 'last':
                    use_dunet = True if i == 0 else False
                    if isinstance(dunet_n_blocks, list):
                        n_blocks = dunet_n_blocks[0]
                    else:
                        n_blocks = dunet_n_blocks
                elif dunet_mode == 'always':
                    use_dunet = True
                    if isinstance(dunet_n_blocks, list):
                        n_blocks = dunet_n_blocks[i]
                    else:
                        n_blocks = dunet_n_blocks
                # DUNet mode 'first'
                else:
                    use_dunet = True if i == len(reversed_f_maps) - 2 else False
                    if isinstance(dunet_n_blocks, list):
                        n_blocks = dunet_n_blocks[len(reversed_f_maps) - 2]
                    else:
                        n_blocks = dunet_n_blocks
            else:
                use_dunet = False
                n_blocks = dunet_n_blocks

            decoder = Decoder(
                in_feature_num,
                out_feature_num,
                basic_module=basic_module,
                conv_layer_order=layer_order,
                conv_kernel_size=conv_kernel_size,
                num_groups=num_groups,
                padding=conv_padding,
                use_dunet=use_dunet,
                dunet_conv_layer_order=dunet_conv_layer_order,
                dunet_n_blocks=n_blocks,
                dunet_num_groups=dunet_num_groups,
                dilated_conv_kernel_size=dilated_conv_kernel_size,
                out_channels_dilated_conv=out_channels_dilated_conv)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid=False,
                 f_maps=64,
                 layer_order='gcr',
                 num_groups=8,
                 num_levels=4,
                 is_segmentation=True,
                 conv_padding=1,
                 **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid=False,
                 f_maps=64,
                 layer_order='gcr',
                 num_groups=8,
                 num_levels=5,
                 is_segmentation=True,
                 conv_padding=1,
                 **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             **kwargs)


class DUNet3D(Abstract3DUNet):
    """
    DUnet3D model, motivated by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6491864/.
    THE IMPLEMENTATION DIFFERS: not all details of the original model
    are publicly available. Here, you can use batchnorm or group norm and an
    activation function inside the dilated network (you can tweak `dunet_conv_layer_order`)
    The original network do not explicitly specify normalization or activation for
    this module. The main difference is that we added a 1x1x1 conv at the end of
    the dilated network, in order to allow both concatenation and summation at
    with the decoder. The input and the output of the dulated Network
    have the same shape.

    Uses DoubleConv as a basic building block, concatenation joining and interpolation
    for upsampling.
    During the joining phase, the information passes through a dilated dense network.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid=False,
                 f_maps=64,
                 layer_order='gcr',
                 num_groups=8,
                 num_levels=4,
                 is_segmentation=True,
                 conv_padding=1,
                 dunet_conv_layer_order='cr',
                 dunet_n_blocks=6,
                 dunet_num_groups=8,
                 dilated_conv_kernel_size=3,
                 out_channels_dilated_conv=32,
                 dunet_mode='last',
                 **kwargs):
        super(DUNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            dunet=True,
            dunet_conv_layer_order=dunet_conv_layer_order,
            dunet_n_blocks=dunet_n_blocks,
            dunet_num_groups=dunet_num_groups,
            dilated_conv_kernel_size=dilated_conv_kernel_size,
            out_channels_dilated_conv=out_channels_dilated_conv,
            dunet_mode=dunet_mode,
            **kwargs)


class ResDUNet3D(Abstract3DUNet):
    """
    ResDUNet3D model, motivated by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6491864/.
    THE IMPLEMENTATION DIFFERS: not all details of the original model
    are publicly available. Here, you can use batchnorm or group norm and an
    activation function inside the dilated network (you can tweak `dunet_conv_layer_order`)
    The original network do not explicitly specify normalization or activation for
    this module. The main difference is that we added a 1x1x1 conv at the end of
    the dilated network, in order to allow both concatenation and summation at
    with the decoder. The input and the output of the dulated Network
    have the same shape.

    Uses DoubleConv as a basic building block, concatenation joining and interpolation
    for upsampling.
    During the joining phase, the information passes through a dilated dense network.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid=True,
                 f_maps=64,
                 layer_order='gcr',
                 num_groups=8,
                 num_levels=4,
                 is_segmentation=True,
                 conv_padding=1,
                 dunet_conv_layer_order='cr',
                 dunet_n_blocks=6,
                 dunet_num_groups=8,
                 dilated_conv_kernel_size=3,
                 out_channels_dilated_conv=32,
                 dunet_mode='last',
                 **kwargs):
        super(ResDUNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ExtResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            dunet=True,
            dunet_conv_layer_order=dunet_conv_layer_order,
            dunet_n_blocks=dunet_n_blocks,
            dunet_num_groups=dunet_num_groups,
            dilated_conv_kernel_size=dilated_conv_kernel_size,
            out_channels_dilated_conv=out_channels_dilated_conv,
            dunet_mode=dunet_mode,
            **kwargs)


class UNet2D(Abstract3DUNet):
    """
    Just a standard 2D Unet. Arises naturally by specifying conv_kernel_size=(1, 3, 3), pool_kernel_size=(1, 2, 2).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 final_sigmoid=False,
                 f_maps=64,
                 layer_order='gcr',
                 num_groups=8,
                 num_levels=4,
                 is_segmentation=True,
                 conv_padding=1,
                 **kwargs):
        if conv_padding == 1:
            conv_padding = (0, 1, 1)
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=(1, 3, 3),
                                     pool_kernel_size=(1, 2, 2),
                                     conv_padding=conv_padding,
                                     **kwargs)


class SegCaps3D(nn.Module):
    """
    Base class for SegCaps3D.

    Args:
        todo.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_caps=1,
                 n_atoms=16,
                 num_levels=4,
                 conv_kernel_size=3,
                 stem_size=3,
                 stem_channels=16,
                 stem_kernel_size=[[1, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
                 stem_dilations=[[1, 1, 1], [6, 1, 1], [12, 1, 1], [18, 1, 1]],
                 stem_order="cr",
                 constrained=True,
                 use_switchnorm=False,
                 segcaps_module=DoubleCaps,
                 segcaps_kernel_size=1,
                 segcaps_num_atoms=32,
                 segcaps_strides=1,
                 segcaps_routings=3,
                 segcaps_sigmoid_routings=True,
                 segcaps_constrained=True,
                 segcaps_use_switchnorm=False,
                 final_decoder_sigmoid=False,
                 union_type="sum",
                 **kwargs):
        super(SegCaps3D, self).__init__()

        if isinstance(n_caps, int):
            n_caps = number_of_features_per_level(n_caps, num_levels=num_levels)
        if isinstance(n_atoms, int):
            n_atoms = number_of_features_per_level(n_atoms,
                                                   num_levels=num_levels)

        # create encoder path consisting of Encoder modules.
        # Depth of the encoder is equal to `len(n_caps)`
        encoders = []
        for i, out_caps_num in enumerate(n_caps):
            if i == 0:
                encoder = StemBlock(n_caps=n_caps[i],
                                    stem_size=stem_size,
                                    in_channels=in_channels,
                                    stem_channels=stem_channels,
                                    stem_kernel_sizes=stem_kernel_size,
                                    stem_order=stem_order,
                                    stem_dilations=stem_dilations)
            else:
                encoder = EncoderCaps(kernel_size=conv_kernel_size,
                                      input_num_capsule=n_caps[i - 1],
                                      input_num_atoms=n_atoms[i - 1],
                                      num_capsule=out_caps_num,
                                      num_atoms=n_atoms[i],
                                      strides=[2, 1],
                                      routings=1,
                                      constrained=constrained,
                                      use_switchnorm=use_switchnorm)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the DecoderCaps modules.
        # The length of the decoder is equal to `len(n_caps) - 1`
        decoders = []
        reversed_n_caps = list(reversed(n_caps))
        reversed_n_atoms = list(reversed(n_atoms))
        # in_caps_nums = list(map(add, reversed_n_caps, ([0] + reversed_n_caps[:-1])))
        for i in range(len(reversed_n_caps) - 1):
            # todo: this is for "cat" union
            in_caps_num = reversed_n_caps[i]
            in_atoms_num = reversed_n_atoms[i]
            out_caps_num = reversed_n_caps[i + 1]
            out_atoms_num = reversed_n_atoms[i + 1]

            decoder = DecoderCaps(kernel_size=conv_kernel_size,
                                  input_num_capsule=in_caps_num,
                                  input_num_atoms=in_atoms_num,
                                  num_capsule=out_caps_num,
                                  num_atoms=out_atoms_num,
                                  strides=[2, 1],
                                  routings=1,
                                  sigmoid_routing=True,
                                  transposed=[True, False],
                                  constrained=constrained,
                                  use_switchnorm=use_switchnorm,
                                  union_type=union_type)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1x1 capsule reduces the number of output
        # channels to the number of labels
        self.segcaps = segcaps_module(kernel_size=segcaps_kernel_size,
                                      input_num_capsule=n_caps[0],
                                      input_num_atoms=n_atoms[0],
                                      num_capsule=out_channels,
                                      num_atoms=segcaps_num_atoms,
                                      strides=segcaps_strides,
                                      routings=segcaps_routings,
                                      sigmoid_routing=segcaps_sigmoid_routings,
                                      constrained=segcaps_constrained,
                                      use_switchnorm=segcaps_use_switchnorm,
                                      activation=squash,
                                      final_squash=False)

        # Length layer to output capsules' predictions
        self.outseg = Length(dim=2, keepdim=False, p=2)

        # Decoder
        decoder = [
            nn.Conv3d(in_channels=segcaps_num_atoms * out_channels,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64,
                      out_channels=128,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=128,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        ]
        if final_decoder_sigmoid:
            decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, y=None):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        segcaps = self.segcaps(x)

        # Computes actual segmentation with vectors' norm
        segmentation = self.outseg(smsquash(segcaps))

        bs, _, _, H, W, D = segcaps.shape

        # For training, the true label is used to mask the output of
        # capsule layer.
        # For prediction, mask using the capsule with maximal length.
        if y is not None:
            # Creating a mask (discarding the background class)
            _, mask = y.max(dim=1, keepdim=True)
            mask = mask.clip(0, 1)
        else:
            # Taking the length of the vector as voxels' classes
            classes = segcaps.norm(dim=2, p=2)
            _, mask = classes.max(dim=1, keepdim=True)

        mask = mask.unsqueeze(2)
        masked = segcaps * mask

        # Merging caps and atoms
        masked = masked.view(bs, -1, H, W, D)

        # Reconstructing via the decoder network
        reconstruction = self.decoder(masked)

        return segmentation, reconstruction


class AttentionSegCaps3D(nn.Module):
    """
    Base class for SegCaps3D.

    Args:
        todo.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_caps=1,
                 n_atoms=16,
                 num_levels=4,
                 conv_kernel_size=3,
                 stem_size=3,
                 stem_channels=16,
                 stem_kernel_size=[[1, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
                 stem_dilations=[[1, 1, 1], [6, 1, 1], [12, 1, 1], [18, 1, 1]],
                 stem_order="cr",
                 constrained=True,
                 use_switchnorm=False,
                 segcaps_module=DoubleCaps,
                 segcaps_kernel_size=1,
                 segcaps_num_atoms=32,
                 segcaps_strides=1,
                 segcaps_routings=3,
                 segcaps_sigmoid_routings=True,
                 segcaps_constrained=True,
                 segcaps_use_switchnorm=False,
                 final_decoder_sigmoid=False,
                 normalization="s",
                 using_bn=False,
                 union_type=None,
                 **kwargs):
        super(AttentionSegCaps3D, self).__init__()

        if isinstance(n_caps, int):
            n_caps = number_of_features_per_level(n_caps, num_levels=num_levels)
        if isinstance(n_atoms, int):
            n_atoms = number_of_features_per_level(n_atoms,
                                                   num_levels=num_levels)

        # create encoder path consisting of Encoder modules.
        # Depth of the encoder is equal to `len(n_caps)`
        encoders = []
        for i, out_caps_num in enumerate(n_caps):
            if i == 0:
                encoder = StemBlock(n_caps=n_caps[i],
                                    stem_size=stem_size,
                                    in_channels=in_channels,
                                    stem_channels=stem_channels,
                                    stem_kernel_sizes=stem_kernel_size,
                                    stem_order=stem_order,
                                    stem_dilations=stem_dilations)
            else:
                encoder = EncoderCaps(kernel_size=conv_kernel_size,
                                      input_num_capsule=n_caps[i - 1],
                                      input_num_atoms=n_atoms[i - 1],
                                      num_capsule=out_caps_num,
                                      num_atoms=n_atoms[i],
                                      strides=[2, 1],
                                      routings=1,
                                      constrained=constrained,
                                      use_switchnorm=use_switchnorm)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the DecoderCaps modules.
        # The length of the decoder is equal to `len(n_caps) - 1`
        decoders = []
        reversed_n_caps = list(reversed(n_caps))
        reversed_n_atoms = list(reversed(n_atoms))
        # in_caps_nums = list(map(add, reversed_n_caps, ([0] + reversed_n_caps[:-1])))
        for i in range(len(reversed_n_caps) - 1):
            # todo: this is for "cat" union
            in_caps_num = reversed_n_caps[i]
            in_atoms_num = reversed_n_atoms[i]
            out_caps_num = reversed_n_caps[i + 1]
            out_atoms_num = reversed_n_atoms[i + 1]

            decoder = AttentionDecoderCaps(kernel_size=conv_kernel_size,
                                           input_num_capsule=in_caps_num,
                                           input_num_atoms=in_atoms_num,
                                           num_capsule=out_caps_num,
                                           num_atoms=out_atoms_num,
                                           strides=[2, 1],
                                           routings=1,
                                           sigmoid_routing=True,
                                           transposed=[True, False],
                                           constrained=constrained,
                                           union_type=union_type,
                                           normalization=normalization,
                                           using_bn=using_bn)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1x1 capsule reduces the number of output
        # channels to the number of labels
        self.segcaps = segcaps_module(kernel_size=segcaps_kernel_size,
                                      input_num_capsule=n_caps[0],
                                      input_num_atoms=n_atoms[0],
                                      num_capsule=out_channels,
                                      num_atoms=segcaps_num_atoms,
                                      strides=segcaps_strides,
                                      routings=segcaps_routings,
                                      sigmoid_routing=segcaps_sigmoid_routings,
                                      constrained=segcaps_constrained,
                                      use_switchnorm=segcaps_use_switchnorm,
                                      activation=squash,
                                      final_squash=False)

        # Length layer to output capsules' predictions
        self.outseg = Length(dim=2, keepdim=False, p=2)

        # Decoder
        decoder = [
            nn.Conv3d(in_channels=segcaps_num_atoms * out_channels,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64,
                      out_channels=128,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=128,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        ]
        if final_decoder_sigmoid:
            decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, y=None):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        segcaps = self.segcaps(x)

        # Computes actual segmentation with vectors' norm
        segmentation = self.outseg(smsquash(segcaps))

        bs, _, _, H, W, D = segcaps.shape

        # For training, the true label is used to mask the output of
        # capsule layer.
        # For prediction, mask using the capsule with maximal length.
        if y is not None:
            # Creating a mask (discarding the background class)
            _, mask = y.max(dim=1, keepdim=True)
            mask = mask.clip(0, 1)
        else:
            # Taking the length of the vector as voxels' classes
            classes = segcaps.norm(dim=2, p=2)
            _, mask = classes.max(dim=1, keepdim=True)

        mask = mask.unsqueeze(2)
        masked = segcaps * mask

        # Merging caps and atoms
        masked = masked.view(bs, -1, H, W, D)

        # Reconstructing via the decoder network
        reconstruction = self.decoder(masked)

        return segmentation, reconstruction


class ClassicSegCaps3D(nn.Module):
    """
    Base class for SegCaps3D.

    Args:
        todo.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_caps=1,
                 n_atoms=16,
                 num_levels=4,
                 conv_kernel_size=3,
                 stem_size=1,
                 stem_channels=128,
                 stem_kernel_size=5,
                 stem_dilation=1,
                 stem_order="cr",
                 constrained=True,
                 use_switchnorm=False,
                 segcaps_module=DoubleCaps,
                 segcaps_kernel_size=1,
                 segcaps_num_atoms=32,
                 segcaps_strides=1,
                 segcaps_routings=3,
                 segcaps_sigmoid_routings=True,
                 segcaps_constrained=True,
                 segcaps_use_switchnorm=False,
                 final_decoder_sigmoid=False,
                 union_type="sum",
                 **kwargs):
        super(ClassicSegCaps3D, self).__init__()

        if isinstance(n_caps, int):
            n_caps = number_of_features_per_level(n_caps, num_levels=num_levels)
        if isinstance(n_atoms, int):
            n_atoms = number_of_features_per_level(n_atoms,
                                                   num_levels=num_levels)

        # create encoder path consisting of Encoder modules.
        # Depth of the encoder is equal to `len(n_caps)`
        encoders = []
        for i, out_caps_num in enumerate(n_caps):
            if i == 0:
                encoder = ClassicStem(num_capsule=n_caps[i],
                                      num_atoms=n_atoms[i],
                                      stem_size=stem_size,
                                      in_channels=in_channels,
                                      stem_channels=stem_channels,
                                      stem_kernel_size=stem_kernel_size,
                                      stem_order=stem_order,
                                      stem_dilation=stem_dilation)
            else:
                encoder = EncoderCaps(kernel_size=conv_kernel_size,
                                      input_num_capsule=n_caps[i - 1],
                                      input_num_atoms=n_atoms[i - 1],
                                      num_capsule=out_caps_num,
                                      num_atoms=n_atoms[i],
                                      strides=[2, 1],
                                      routings=1,
                                      constrained=constrained,
                                      use_switchnorm=use_switchnorm)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the DecoderCaps modules.
        # The length of the decoder is equal to `len(n_caps) - 1`
        decoders = []
        reversed_n_caps = list(reversed(n_caps))
        reversed_n_atoms = list(reversed(n_atoms))
        # in_caps_nums = list(map(add, reversed_n_caps, ([0] + reversed_n_caps[:-1])))
        for i in range(len(reversed_n_caps) - 1):
            # todo: this is for "cat" union
            in_caps_num = reversed_n_caps[i]
            in_atoms_num = reversed_n_atoms[i]
            out_caps_num = reversed_n_caps[i + 1]
            out_atoms_num = reversed_n_atoms[i + 1]

            decoder = DecoderCaps(kernel_size=conv_kernel_size,
                                  input_num_capsule=in_caps_num,
                                  input_num_atoms=in_atoms_num,
                                  num_capsule=out_caps_num,
                                  num_atoms=out_atoms_num,
                                  strides=[2, 1],
                                  routings=1,
                                  sigmoid_routing=True,
                                  transposed=[True, False],
                                  constrained=constrained,
                                  use_switchnorm=use_switchnorm,
                                  union_type=union_type)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1x1 capsule reduces the number of output
        # channels to the number of labels
        self.segcaps = segcaps_module(kernel_size=segcaps_kernel_size,
                                      input_num_capsule=n_caps[0],
                                      input_num_atoms=n_atoms[0],
                                      num_capsule=out_channels,
                                      num_atoms=segcaps_num_atoms,
                                      strides=segcaps_strides,
                                      routings=segcaps_routings,
                                      sigmoid_routing=segcaps_sigmoid_routings,
                                      constrained=segcaps_constrained,
                                      use_switchnorm=segcaps_use_switchnorm,
                                      activation=squash,
                                      final_squash=False)

        # Length layer to output capsules' predictions
        self.outseg = Length(dim=2, keepdim=False, p=2)

        # Decoder
        decoder = [
            nn.Conv3d(in_channels=segcaps_num_atoms * out_channels,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64,
                      out_channels=128,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=128,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        ]
        if final_decoder_sigmoid:
            decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, y=None):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        segcaps = self.segcaps(x)

        # Computes actual segmentation with vectors' norm
        segmentation = self.outseg(smsquash(segcaps))

        bs, _, _, H, W, D = segcaps.shape

        # For training, the true label is used to mask the output of
        # capsule layer.
        # For prediction, mask using the capsule with maximal length.
        if y is not None:
            # Creating a mask (discarding the background class)
            _, mask = y.max(dim=1, keepdim=True)
            mask = mask.clip(0, 1)
        else:
            # Taking the length of the vector as voxels' classes
            classes = segcaps.norm(dim=2, p=2)
            _, mask = classes.max(dim=1, keepdim=True)

        mask = mask.unsqueeze(2)
        masked = segcaps * mask

        # Merging caps and atoms
        masked = masked.view(bs, -1, H, W, D)

        # Reconstructing via the decoder network
        reconstruction = self.decoder(masked)

        return segmentation, reconstruction


class ClassicAttentionSegCaps3D(nn.Module):
    """
    Base class for SegCaps3D.

    Args:
        todo.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_caps=1,
                 n_atoms=16,
                 num_levels=4,
                 conv_kernel_size=3,
                 stem_size=1,
                 stem_channels=128,
                 stem_kernel_size=5,
                 stem_dilation=1,
                 stem_order="cr",
                 constrained=True,
                 use_switchnorm=False,
                 segcaps_module=DoubleCaps,
                 segcaps_kernel_size=1,
                 segcaps_num_atoms=32,
                 segcaps_strides=1,
                 segcaps_routings=3,
                 segcaps_sigmoid_routings=True,
                 segcaps_constrained=True,
                 segcaps_use_switchnorm=False,
                 final_decoder_sigmoid=False,
                 normalization="s",
                 using_bn=False,
                 union_type=None,
                 **kwargs):
        super(ClassicAttentionSegCaps3D, self).__init__()

        if isinstance(n_caps, int):
            n_caps = number_of_features_per_level(n_caps, num_levels=num_levels)
        if isinstance(n_atoms, int):
            n_atoms = number_of_features_per_level(n_atoms,
                                                   num_levels=num_levels)

        # create encoder path consisting of Encoder modules.
        # Depth of the encoder is equal to `len(n_caps)`
        encoders = []
        for i, out_caps_num in enumerate(n_caps):
            if i == 0:
                encoder = ClassicStem(num_capsule=n_caps[i],
                                      num_atoms=n_atoms[i],
                                      stem_size=stem_size,
                                      in_channels=in_channels,
                                      stem_channels=stem_channels,
                                      stem_kernel_size=stem_kernel_size,
                                      stem_order=stem_order,
                                      stem_dilation=stem_dilation)
            else:
                encoder = EncoderCaps(kernel_size=conv_kernel_size,
                                      input_num_capsule=n_caps[i - 1],
                                      input_num_atoms=n_atoms[i - 1],
                                      num_capsule=out_caps_num,
                                      num_atoms=n_atoms[i],
                                      strides=[2, 1],
                                      routings=1,
                                      constrained=constrained,
                                      use_switchnorm=use_switchnorm)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the DecoderCaps modules.
        # The length of the decoder is equal to `len(n_caps) - 1`
        decoders = []
        reversed_n_caps = list(reversed(n_caps))
        reversed_n_atoms = list(reversed(n_atoms))
        # in_caps_nums = list(map(add, reversed_n_caps, ([0] + reversed_n_caps[:-1])))
        for i in range(len(reversed_n_caps) - 1):
            # todo: this is for "cat" union
            in_caps_num = reversed_n_caps[i]
            in_atoms_num = reversed_n_atoms[i]
            out_caps_num = reversed_n_caps[i + 1]
            out_atoms_num = reversed_n_atoms[i + 1]

            decoder = AttentionDecoderCaps(kernel_size=conv_kernel_size,
                                           input_num_capsule=in_caps_num,
                                           input_num_atoms=in_atoms_num,
                                           num_capsule=out_caps_num,
                                           num_atoms=out_atoms_num,
                                           strides=[2, 1],
                                           routings=1,
                                           sigmoid_routing=True,
                                           transposed=[True, False],
                                           constrained=constrained,
                                           union_type=union_type,
                                           normalization=normalization,
                                           using_bn=using_bn)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1x1 capsule reduces the number of output
        # channels to the number of labels
        self.segcaps = segcaps_module(kernel_size=segcaps_kernel_size,
                                      input_num_capsule=n_caps[0],
                                      input_num_atoms=n_atoms[0],
                                      num_capsule=out_channels,
                                      num_atoms=segcaps_num_atoms,
                                      strides=segcaps_strides,
                                      routings=segcaps_routings,
                                      sigmoid_routing=segcaps_sigmoid_routings,
                                      constrained=segcaps_constrained,
                                      use_switchnorm=segcaps_use_switchnorm,
                                      activation=squash,
                                      final_squash=False)

        # Length layer to output capsules' predictions
        self.outseg = Length(dim=2, keepdim=False, p=2)

        # Decoder
        decoder = [
            nn.Conv3d(in_channels=segcaps_num_atoms * out_channels,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64,
                      out_channels=128,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=128,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        ]
        if final_decoder_sigmoid:
            decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, y=None):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        segcaps = self.segcaps(x)

        # Computes actual segmentation with vectors' norm
        segmentation = self.outseg(smsquash(segcaps))

        bs, _, _, H, W, D = segcaps.shape

        # For training, the true label is used to mask the output of
        # capsule layer.
        # For prediction, mask using the capsule with maximal length.
        if y is not None:
            # Creating a mask (discarding the background class)
            _, mask = y.max(dim=1, keepdim=True)
            mask = mask.clip(0, 1)
        else:
            # Taking the length of the vector as voxels' classes
            classes = segcaps.norm(dim=2, p=2)
            _, mask = classes.max(dim=1, keepdim=True)

        mask = mask.unsqueeze(2)
        masked = segcaps * mask

        # Merging caps and atoms
        masked = masked.view(bs, -1, H, W, D)

        # Reconstructing via the decoder network
        reconstruction = self.decoder(masked)

        return segmentation, reconstruction


class RSCaps3D(nn.Module):

    def __init__(self,
                 stem_size=1,
                 input_size=64,
                 n_class=3,
                 in_channels=1,
                 stem_n_caps=1,
                 stem_channels=128,
                 stem_kernel_size=[[1, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
                 stem_dilations=[[1, 1, 1], [6, 1, 1], [12, 1, 1], [18, 1, 1]],
                 stem_order="cr",
                 num_residual_blocks=2,
                 residual_size=3,
                 residual_kernel_size=3,
                 residual_num_capsules=4,
                 residual_num_atoms=16,
                 residual_strides=1,
                 residual_routings=1,
                 residual_sigmoid_routing=True,
                 residual_transposed=False,
                 residual_constrained=True,
                 residual_skip_capsule=False,
                 residual_skip_kernel_size=1,
                 residual_skip_strides=1,
                 residual_skip_num_capsule=4,
                 residual_skip_routings=1,
                 residual_skip_sigmoid_routing=True,
                 residual_skip_constrained=True,
                 residual_union_type="cat",
                 residual_squash_sum_union=False,
                 segcaps_kernel_size=1,
                 segcaps_num_atoms=32,
                 segcaps_strides=1,
                 segcaps_routings=3,
                 segcaps_sigmoid_routings=True,
                 segcaps_constrained=True,
                 final_decoder_sigmoid=False,
                 **kwargs):
        super(RSCaps3D, self).__init__()

        # Stem layer
        self.stem = StemBlock(n_caps=stem_n_caps,
                              stem_size=stem_size,
                              in_channels=in_channels,
                              stem_channels=stem_channels,
                              stem_kernel_sizes=stem_kernel_size,
                              stem_order=stem_order,
                              stem_dilations=stem_dilations)

        # Residual Blocks
        depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1

        def _c(v, n):
            if type(v) != list:
                return [v] * n
            else:
                return v

        residual_size = _c(residual_size, num_residual_blocks)
        residual_kernel_size = _c(residual_kernel_size, num_residual_blocks)
        residual_num_capsules = _c(residual_num_capsules, num_residual_blocks)
        residual_num_atoms = _c(residual_num_atoms, num_residual_blocks)
        residual_strides = _c(residual_strides, num_residual_blocks)
        residual_routings = _c(residual_routings, num_residual_blocks)
        residual_sigmoid_routing = _c(residual_sigmoid_routing,
                                      num_residual_blocks)
        residual_transposed = _c(residual_transposed, num_residual_blocks)
        residual_constrained = _c(residual_constrained, num_residual_blocks)
        residual_skip_capsule = _c(residual_skip_capsule, num_residual_blocks)
        residual_skip_kernel_size = _c(residual_skip_kernel_size,
                                       num_residual_blocks)
        residual_skip_strides = _c(residual_skip_strides, num_residual_blocks)
        residual_skip_num_capsule = _c(residual_skip_num_capsule,
                                       num_residual_blocks)
        residual_skip_routings = _c(residual_skip_routings, num_residual_blocks)
        residual_skip_sigmoid_routing = _c(residual_skip_sigmoid_routing,
                                           num_residual_blocks)
        residual_skip_constrained = _c(residual_skip_constrained,
                                       num_residual_blocks)

        if residual_union_type == "cat":
            if depth(residual_num_capsules) == 1:
                input_num_capsule = [stem_n_caps] + (
                    [residual_num_capsules[-1] * 2] * num_residual_blocks)
            elif depth(residual_num_capsules) == 2:
                r = [c[0] + c[-1] for c in residual_num_capsules]
                input_num_capsule = [stem_n_caps] + r
        elif residual_union_type == "sum":
            if depth(residual_num_capsules) == 1:
                input_num_capsule = [stem_n_caps] + (
                    [residual_num_capsules[-1]] * num_residual_blocks)
            elif depth(residual_num_capsules) == 2:
                r = [c[-1] for c in residual_num_capsules]
                input_num_capsule = [stem_n_caps] + r

        if depth(residual_num_atoms) == 1:
            input_num_atoms = [stem_channels] + ([residual_num_atoms[-1]] *
                                                 num_residual_blocks)
        elif depth(residual_num_atoms) == 2:
            r = [c[-1] for c in residual_num_atoms]
            input_num_atoms = [stem_channels] + r

        blocks = [
            ResCapsBlock(block_size=residual_size[i],
                         kernel_size=residual_kernel_size[i],
                         input_num_capsule=input_num_capsule[i],
                         input_num_atoms=input_num_atoms[i],
                         num_capsule=residual_num_capsules[i],
                         num_atoms=residual_num_atoms[i],
                         strides=residual_strides[i],
                         routings=residual_routings[i],
                         sigmoid_routing=residual_sigmoid_routing[i],
                         transposed=residual_transposed[i],
                         constrained=residual_constrained[i],
                         skip_capsule=residual_skip_capsule[i],
                         skip_kernel_size=residual_skip_kernel_size[i],
                         skip_strides=residual_skip_strides[i],
                         skip_num_capsule=residual_skip_num_capsule[i],
                         skip_routings=residual_skip_routings[i],
                         skip_sigmoid_routing=residual_skip_sigmoid_routing[i],
                         skip_constrained=residual_skip_constrained[i],
                         union_type=residual_union_type,
                         squash_sum_union=residual_squash_sum_union,
                         input_size=input_size)
            for i in range(num_residual_blocks)
        ]
        self.residual_blocks = nn.Sequential(*blocks)

        # Segmentation Capsule
        p, _ = calc_same_padding(input_size,
                                 kernel=segcaps_kernel_size,
                                 stride=segcaps_strides)
        self.segcaps = SingleCaps(kernel_size=segcaps_kernel_size,
                                  input_num_capsule=input_num_capsule[-1],
                                  input_num_atoms=input_num_atoms[-1],
                                  num_capsule=n_class,
                                  num_atoms=segcaps_num_atoms,
                                  strides=segcaps_strides,
                                  routings=segcaps_routings,
                                  sigmoid_routing=segcaps_sigmoid_routings,
                                  constrained=segcaps_constrained)

        # Length layer to output capsules' predictions
        self.outseg = Length(dim=2, keepdim=False, p=1)

        # Decoder
        decoder = [
            nn.Conv3d(in_channels=segcaps_num_atoms * n_class,
                      out_channels=64,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64,
                      out_channels=128,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=128,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
        ]
        if final_decoder_sigmoid:
            decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, y=None):
        # Stem layer
        x = self.stem(x)
        # Residual Blocks
        x = self.residual_blocks(x)
        # Segmentation Capsule
        segcaps = self.segcaps(x)

        # Computes actual segmentation with vectors' norm
        segmentation = self.outseg(segcaps)

        bs, _, _, H, W, D = segcaps.shape

        # For training, the true label is used to mask the output of
        # capsule layer.
        # For prediction, mask using the capsule with maximal length.
        if y is not None:
            # Creating a mask (discarding the background class)
            _, mask = y.max(dim=1, keepdim=True)
            mask = mask.clip(0, 1)
        else:
            # Taking the length of the vector as voxels' classes
            classes = segcaps.norm(dim=2, p=1)
            _, mask = classes.max(dim=1, keepdim=True)

        mask = mask.unsqueeze(2)
        masked = segcaps * mask

        # Merging caps and atoms
        masked = masked.view(bs, -1, H, W, D)

        # Reconstructing via the decoder network
        reconstruction = self.decoder(masked)

        return segmentation, reconstruction


class SegmentationModel(pl.LightningModule):

    def __init__(self,
                 hparams,
                 seg_loss=FocalTversky_loss({"apply_nonlin": nn.Softmax(dim=1)
                                            }),
                 rec_loss=F.mse_loss,
                 optimizer=optim.AdamW,
                 scheduler=None,
                 learning_rate=1e-3,
                 classes_names=None,
                 is_capsnet=False):
        super(SegmentationModel, self).__init__()
        self.hp = hparams
        self.learning_rate = learning_rate
        self.seg_loss = seg_loss
        self.rec_loss = rec_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_capsnet = is_capsnet
        if classes_names:
            assert len(classes_names) == hparams['out_channels']
        self.classes_names = classes_names

        if hparams['modeltype'] == 'UNet3D':
            self._model = UNet3D(final_sigmoid=False, **hparams)

        elif hparams['modeltype'] == 'ResidualUNet3D':
            self._model = ResidualUNet3D(final_sigmoid=False, **hparams)

        elif hparams['modeltype'] == 'DUNet3D':
            self._model = DUNet3D(final_sigmoid=False, **hparams)

        elif hparams['modeltype'] == 'ResDUNet3D':
            self._model = ResDUNet3D(final_sigmoid=False, **hparams)

        elif hparams['modeltype'] == 'RSCaps3D':
            self._model = RSCaps3D(**hparams)

        elif hparams['modeltype'] == 'SegCaps3D':
            self._model = SegCaps3D(**hparams)

        elif hparams['modeltype'] == 'ClassicSegCaps3D':
            self._model = ClassicSegCaps3D(**hparams)

        elif hparams['modeltype'] == 'AttentionSegCaps3D':
            self._model = AttentionSegCaps3D(**hparams)

        elif hparams['modeltype'] == 'ClassicAttentionSegCaps3D':
            self._model = ClassicAttentionSegCaps3D(**hparams)

        else:
            raise ValueError(f"Unknown model: {hparams['modeltype']}")

        # Configure metrics
        # metrics = [
        #     metric.DiceCoefficient(),
        #     metric.VolumeSimilarity(),
        #     metric.MahalanobisDistance()
        # ]
        # self.evaluator = eval_.SegmentationEvaluator(metrics, labels)

    def forward(self, x, y=None):
        if self.is_capsnet:
            return self._model(x, y)
        else:
            return self._model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        optimizers = {"optimizer": optimizer}

        if self.scheduler:
            optimizers["lr_scheduler"] = self.scheduler(optimizer)
            optimizers["monitor"] = "Validation SegLoss"

        return optimizers

    def step(self, batch, batch_idx, step_name="Training"):
        x, y = batch["mri"]["data"], batch["label"]["data"]
        _, labels = y.max(dim=1)

        names = {v[0]: k for k, v in batch["labels_names"].items()}
        head = names.get("HEAD", -1)
        tail = names.get("TAIL", -2)

        if self.is_capsnet:
            y_hat, reconstruction = self.forward(x, y)
            mask = labels.unsqueeze(1).clip(0, 1)
            target = x * mask
            rec_loss = self.rec_loss(reconstruction, target)
            # seg_loss = self.seg_loss(y_hat, labels.long())
            seg_loss = forgiving_loss(self.seg_loss,
                                      y_hat,
                                      labels.long(),
                                      batch["ca_type"][0],
                                      head=head,
                                      tail=tail)
            loss = seg_loss + self.hp['α'] * rec_loss
            # self.log(f"{step_name} Segmentation Loss", seg_loss)
            self.logger.experiment.log_metric(f"{step_name} Segmentation Loss",
                                              seg_loss)
            # self.log(f"{step_name} Reconstruction Loss", rec_loss)
            self.logger.experiment.log_metric(
                f"{step_name} Reconstruction Loss", rec_loss)
        else:
            y_hat = self.forward(x.float())
            # loss = self.seg_loss(y_hat, labels.long())
            loss = forgiving_loss(self.seg_loss,
                                  y_hat,
                                  labels.long(),
                                  batch["ca_type"][0],
                                  head=head,
                                  tail=tail)
            # self.log(f"{step_name} Segmentation Loss", loss)
            self.logger.experiment.log_metric(f"{step_name} Segmentation Loss",
                                              loss)

        if (batch_idx == 0) and step_name == "Training":
            self.logger.experiment.set_model_graph(self._model, overwrite=True)

        if (batch_idx in [0, 1]) and (step_name == "Test"):
            middle_slice = x.shape[-2] // 2
            mri = x[0, 0, :, middle_slice, :].numpy()
            manual_seg = labels[0, :, middle_slice, :].numpy()
            _, output_seg = y_hat.max(dim=1)
            output_seg = output_seg[0, :, middle_slice, :].numpy()
            self.logger.experiment.log_image(mri, name=f"MRI: {batch_idx}")
            self.logger.experiment.log_image(
                manual_seg, name=f"Manual Segmentation: {batch_idx}")
            self.logger.experiment.log_image(
                output_seg, name=f"Predicted Segmentation: {batch_idx}")

        # if step_name == "Test":
        #     labels = {k: v[0] for k, v in batch["labels_names"].items()}
        #     evaluator.evaluate(batch["label"]["data"], batch["label"]["data"] * 0.1,
        #            batch["label"]["stem"][0])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, step_name="Training")

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, step_name="Validation")

        # # Specific metrics
        # y_hat = F.softmax(y_hat, dim=1)

        # dicecoef = metrics.DiceCoefficient()
        # meaniou = metrics.MeanIoU()

        # per_channel_dice = dicecoef(y_hat, y.float(), per_channel=True)
        # per_channel_iou = meaniou(y_hat, y.float(), per_channel=True)

        # assert len(per_channel_iou) == len(
        #     per_channel_dice) == self.hp['out_channels']

        # # Let's assume the first value is background
        # self.log("Validation Dice", torch.mean(per_channel_dice[1:]))
        # self.log("Validation IoU", torch.mean(per_channel_iou[1:]))

        # for i in range(self.hp['out_channels']):
        #     name = self.classes_names[i] if self.classes_names else i
        #     self.log(f"Validation Dice {name}", per_channel_dice[i])
        #     self.log(f"Validation IoU {name}", per_channel_iou[i])

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, step_name="Test")

        return loss
