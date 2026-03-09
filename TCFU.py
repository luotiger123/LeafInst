class MaskFeatModule(BaseModule):
    """Mask feature branch for CondInst-style dynamic mask prediction.

    This module is adapted from the original mask feature branch used in
    CondInst / YOLACT-style instance segmentation frameworks, with a modified
    top-down feature fusion strategy.

    Compared with the conventional top-down additive fusion approach (TAFU),
    this implementation adopts the proposed Top-down Concatenation-decoder
    Feature Fusion (TCFU). Specifically, higher-level features are first
    upsampled, then concatenated with lower-level features, and finally
    compressed by a 1x1 convolution for channel reduction. This design is
    intended to alleviate redundant high-frequency feature stacking and
    improve high-resolution mask feature representation.

    Args:
        in_channels (int): Number of channels in the input feature maps.
        feat_channels (int): Number of hidden channels in the mask feature
            branch.
        start_level (int): First FPN level used to generate mask features.
        end_level (int): Last FPN level used to generate mask features.
        out_channels (int): Number of output channels in the predicted mask
            feature map. This is the channel dimension dynamically convolved
            with the predicted kernels.
        mask_stride (int): Downsample factor of the output mask feature map.
            Defaults to 4.
        num_stacked_convs (int): Number of stacked convolution layers in the
            mask feature branch.
        conv_cfg (dict, optional): Config dict for convolution layers.
            Defaults to None.
        norm_cfg (dict, optional): Config dict for normalization layers.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization config.
    """

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 start_level: int,
                 end_level: int,
                 out_channels: int,
                 mask_stride: int = 4,
                 num_stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = [
                     dict(type='Normal', layer='Conv2d', std=0.01)
                 ],
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        self.mask_stride = mask_stride
        self.num_stacked_convs = num_stacked_convs
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        assert start_level >= 0 and end_level >= start_level

        self._init_layers()

        # TCFU stage 1:
        # Upsample higher-level feature, then concatenate with the adjacent
        # lower-level feature, followed by 1x1 channel compression.
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.downc1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # TCFU stage 2:
        # Repeat the same concatenation-decoder fusion process for the next
        # lower level.
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.downc2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def _init_layers(self) -> None:
        """Initialize convolution layers for multi-level mask feature extraction."""
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            convs_per_level.add_module(
                f'conv{i}',
                ConvModule(
                    self.in_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False,
                    bias=False))
            self.convs_all_levels.append(convs_per_level)

        conv_branch = []
        for _ in range(self.num_stacked_convs):
            conv_branch.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=False))
        self.conv_branch = nn.Sequential(*conv_branch)

        self.conv_pred = nn.Conv2d(
            self.feat_channels, self.out_channels, 1, stride=1)

    def init_weights(self) -> None:
        """Initialize weights of the module."""
        super().init_weights()
        kaiming_init(self.convs_all_levels, a=1, distribution='uniform')
        kaiming_init(self.conv_branch, a=1, distribution='uniform')
        kaiming_init(self.conv_pred, a=1, distribution='uniform')

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        """Forward features from the upstream FPN.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream network.

        Returns:
            Tensor: Predicted mask feature map.
        """
        inputs = x[self.start_level:self.end_level + 1]
        inputs = list(inputs)
        assert len(inputs) == (self.end_level - self.start_level + 1)

        # Project all input levels to the same channel dimension.
        for i in range(0, 3):
            inputs[i] = self.convs_all_levels[i](inputs[i])

        # TCFU:
        # Upsample higher-level features, concatenate with lower-level features,
        # and reduce channels via 1x1 convolution.
        for i in range(1, 3):
            up_layer = getattr(self, f"up{i}")
            inputs[3 - i] = up_layer(inputs[3 - i])

            down_layer = getattr(self, f"downc{i}")
            inputs[3 - i - 1] = down_layer(
                torch.cat([inputs[3 - i], inputs[3 - i - 1]], dim=1)
            )

        feature_add_all_level = self.conv_branch(inputs[0])
        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred