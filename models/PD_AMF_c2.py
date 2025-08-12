import torch
import torch.nn as nn
import torch.nn.functional as F



def norm3d(channels):
    """恢复为3D Batch Normalization"""
    return nn.BatchNorm3d(channels)


def conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
    """3D卷积层"""
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)

class DropBlock3D(nn.Module):
    """3D DropBlock正则化"""
    def __init__(self, block_size=5, drop_prob=0.05):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        B, C, D, H, W = x.shape
        # 确保block_size不超过特征图尺寸
        actual_block_size = min(self.block_size, min(D, H, W))

        # 计算gamma值
        gamma = (self.drop_prob * D * H * W) / (actual_block_size ** 3)

        # 避免除零错误
        valid_regions = max(1, (D - actual_block_size + 1) * (H - actual_block_size + 1) * (W - actual_block_size + 1))
        gamma = gamma / valid_regions

        # 生成随机mask
        mask = torch.rand(B, 1, D, H, W, device=x.device) < gamma
        mask = mask.float()

        # 扩展mask到block size
        if actual_block_size > 1:
            mask = torch.nn.functional.max_pool3d(mask, actual_block_size, stride=1, padding=actual_block_size // 2)
        mask = 1 - mask

        # 应用mask并标准化
        x = x * mask
        scale_factor = mask.numel() / (mask.sum() + 1e-7)  # 避免除零
        x = x * scale_factor

        return x


# ===================== 注意力模块 =====================

class SELayer3D(nn.Module):
    """3D Squeeze-and-Excitation层"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(4, channels // reduction)  # 动态避免过度压缩
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),  # 添加dropout提高鲁棒性
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1, 1)
        return x * y
class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, stride=1, padding=0),
            nn.ELU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out * x


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out) * x


class CBAM3D(nn.Module):
    """3D CBAM模块：结合通道注意力和空间注意力"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# ===================== 基础模块 =====================
class SENetplusBottleneck(nn.Module):          #CBAMBottleneck

    def __init__(self, in_channels, out_channels, cardinality=8, stride=1, expansion=2, dilation=1):
        super().__init__()
        self.expansion = expansion

        # 1x1 conv
        self.conv1 = conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = norm3d(out_channels)

        # 3x3 grouped conv
        self.conv2 = conv3d(out_channels, out_channels, kernel_size=3, stride=stride,
                                padding=dilation, dilation=dilation, groups=cardinality)
        self.bn2 = norm3d(out_channels)

        # 1x1 conv (expansion)
        self.conv3 = conv3d(out_channels, out_channels * expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = norm3d(out_channels * expansion)

        # SE attention
        self.se = SELayer3D(out_channels * expansion)
        self.cbam = CBAM3D(out_channels * expansion)

        # 使用ELU激活函数，支持负值且平滑
        self.activation = nn.ELU(inplace=True)

        # 残差连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels * expansion:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, padding=0),
                norm3d(out_channels * expansion)
            )

    def forward(self, x):
        identity = x

        # 主分支
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out = self.se(out)
        out = self.cbam(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.activation(out)

        return out


class SENetplus_DilatedBottleneck(nn.Module):       #CBAMDilatedBottleneck
    """QSM优化的膨胀卷积Bottleneck块"""
    def __init__(self, in_channels, out_channels, cardinality=8, stride=1, expansion=2):
        super().__init__()
        self.expansion = expansion

        # 1x1 conv
        self.conv1 = conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = norm3d(out_channels)

        # 3x3 dilated grouped conv
        self.conv2 = conv3d(out_channels, out_channels, kernel_size=3, stride=stride,
                            padding=2, dilation=2, groups=cardinality)
        self.bn2 = norm3d(out_channels)

        # 1x1 conv (expansion)
        self.conv3 = conv3d(out_channels, out_channels * expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = norm3d(out_channels * expansion)

        #  attention
        self.se = SELayer3D(out_channels * expansion)
        self.cbam = CBAM3D(out_channels * expansion)

        # 使用ELU激活函数
        self.activation = nn.ELU(inplace=True)

        # 残差连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels * expansion:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, padding=0),
                norm3d(out_channels * expansion)
            )

    def forward(self, x):
        identity = x

        # 主分支
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.cbam(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.activation(out)

        return out

# ===================== 多模态融合模块 =====================

class AdaptiveMultimodalFusion(nn.Module):           #AMF block
    """自适应多模态融合模块"""

    def __init__(self, channels):
        super().__init__()
        # 为每个模态学习注意力权重
        self.conv1 = conv3d(channels * 3, channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = norm3d(channels)

        self.conv2 = conv3d(channels * 3, channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = norm3d(channels)

        self.conv3 = conv3d(channels * 3, channels, kernel_size=3, stride=1, padding=1)
        self.gn3 = norm3d(channels)

        # 特征融合后的后处理
        self.fusion_conv = conv3d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.fusion_gn = norm3d(channels)
        self.activation = nn.ELU(inplace=True)

    def forward(self, qsm_feature, roi_feature, t1_feature):
        # 连接所有模态特征
        concatenated = torch.cat([qsm_feature, roi_feature, t1_feature], dim=1)

        # 为每个模态计算注意力权重
        qsm_attention = torch.sigmoid(self.gn1(self.conv1(concatenated)))
        roi_attention = torch.sigmoid(self.gn2(self.conv2(concatenated)))
        t1_attention = torch.sigmoid(self.gn3(self.conv3(concatenated)))

        # 归一化注意力权重
        total_attention = qsm_attention + roi_attention + t1_attention + 1e-8
        qsm_attention = qsm_attention / total_attention
        roi_attention = roi_attention / total_attention
        t1_attention = t1_attention / total_attention

        # 加权融合
        fused_feature = (qsm_attention * qsm_feature +
                         roi_attention * roi_feature +
                         t1_attention * t1_feature)

        # 后处理
        fused_feature = self.activation(self.fusion_gn(self.fusion_conv(fused_feature)))

        return fused_feature


# ===================== 特征提取模块 =====================

class FeatureExtractor(nn.Module):       #Stem
    """特征提取stem - 针对QSM数据优化"""

    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.stem = nn.Sequential(
            conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm3d(out_channels),
            nn.ELU(inplace=True),

            conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm3d(out_channels),
            nn.ELU(inplace=True),

            conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            norm3d(out_channels),
            nn.ELU(inplace=True),

            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.stem(x)

# ===================== 主网络 =====================

class PD_AMFNet_Gated(nn.Module):       #GatefuseNet

    def __init__(self,
                 layers=[3, 4, 6, 3],  # 每层的块数
                 base_width=16,  # 基础通道数
                 cardinality=8,  # 分组卷积数
                 expansion=2,  # 扩展比例
                 num_classes=1,  # 输出类别数
                 drop_prob=0.05):  # DropBlock概率

        super().__init__()
        self.expansion = expansion

        # 计算各层通道数
        channels = [base_width * (2 ** i) for i in range(5)]  # [16, 32, 64, 128, 256]

        # 特征提取器
        self.qsm_extractor = FeatureExtractor(in_channels=1, out_channels=base_width)
        self.roi_extractor = FeatureExtractor(in_channels=1, out_channels=base_width)
        self.t1_extractor = FeatureExtractor(in_channels=1, out_channels=base_width)

        # 融合模块
        self.amf_modules = nn.ModuleList([
            AdaptiveMultimodalFusion(base_width),
            AdaptiveMultimodalFusion(channels[1]),
            AdaptiveMultimodalFusion(channels[2]),
            AdaptiveMultimodalFusion(channels[3])
        ])

        # 细粒度 gate (channel‑wise)  [stage0,1,2,3]
        gate_channels = [
            base_width,
            channels[0] * expansion,
            channels[1] * expansion,
            channels[2] * expansion,
        ]
        # self.gate_params = nn.ParameterList([nn.Parameter(torch.zeros(c)) for c in gate_channels])
        self.gate_params = nn.ParameterList([
            nn.Parameter(torch.full((c,), 0.5))  # 初始化为-1，sigmoid后约为0.27
            for c in gate_channels
        ])


        # 跟踪各路径的输入通道数
        self.main_in_channels = base_width
        self.roi_in_channels = base_width
        self.t1_in_channels = base_width

        # 主干网络层
        self.layer1 = self._make_layer(SENetplusBottleneck, channels[0], layers[0], cardinality, stride=1, path='main')
        self.layer2 = self._make_layer(SENetplusBottleneck, channels[1], layers[1], cardinality, stride=2, path='main')
        self.layer3 = self._make_layer(SENetplusBottleneck, channels[2], layers[2], cardinality, stride=2, path='main')
        self.layer4 = self._make_layer(SENetplus_DilatedBottleneck, channels[3], layers[3], cardinality, stride=1, path='main')

        # ROI路径层
        self.roi_layer1 = self._make_layer(SENetplusBottleneck, channels[0], layers[0], cardinality, stride=1, path='roi')
        self.roi_layer2 = self._make_layer(SENetplusBottleneck, channels[1], layers[1], cardinality, stride=2, path='roi')
        self.roi_layer3 = self._make_layer(SENetplusBottleneck, channels[2], layers[2], cardinality, stride=2, path='roi')

        # T1路径层
        self.t1_layer1 = self._make_layer(SENetplusBottleneck, channels[0], layers[0], cardinality, stride=1, path='t1')
        self.t1_layer2 = self._make_layer(SENetplusBottleneck, channels[1], layers[1], cardinality, stride=2, path='t1')
        self.t1_layer3 = self._make_layer(SENetplusBottleneck, channels[2], layers[2], cardinality, stride=2, path='t1')

        # 分类头
        self.dropblock = DropBlock3D(block_size=5, drop_prob=drop_prob)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(channels[3] * expansion, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, cardinality, stride=1, path='main'):
        """构建网络层"""
        layers = []

        # 获取对应路径的输入通道数
        if path == 'main':
            in_channels = self.main_in_channels
        elif path == 'roi':
            in_channels = self.roi_in_channels
        else:  # t1
            in_channels = self.t1_in_channels

        # 第一个块
        layers.append(block(in_channels, out_channels, cardinality, stride, self.expansion))
        in_channels = out_channels * self.expansion

        # 剩余块
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels, cardinality, 1, self.expansion))
            in_channels = out_channels * self.expansion

        # 更新对应路径的输入通道数
        if path == 'main':
            self.main_in_channels = in_channels
        elif path == 'roi':
            self.roi_in_channels = in_channels
        else:  # t1
            self.t1_in_channels = in_channels

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:  # 添加bias检查
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:  # 添加bias检查
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # 添加bias检查
                    nn.init.constant_(m.bias, 0)

    def forward(self, qsm_input, roi_input, t1_input):
        """前向传播"""
        # Stage 0: 特征提取
        qsm_feat = self.qsm_extractor(qsm_input)
        roi_feat = self.roi_extractor(roi_input)
        t1_feat = self.t1_extractor(t1_input)

        # Stage 0: 门控融合
        fusion_0 = self.amf_modules[0](qsm_feat, roi_feat, t1_feat)
        # main_feat = qsm_feat + torch.sigmoid(self.gate_params[0]) * fusion_0
        gate0 = torch.sigmoid(self.gate_params[0]).view(1, -1, 1, 1, 1)
        main_feat = qsm_feat + gate0 * fusion_0

        # Stage 1: 处理各路径
        main_1 = self.layer1(main_feat)
        roi_1 = self.roi_layer1(roi_feat)
        t1_1 = self.t1_layer1(t1_feat)

        # Stage 1: 门控融合
        fusion_1 = self.amf_modules[1](main_1, roi_1, t1_1)
        # main_1 = main_1 + torch.sigmoid(self.gate_params[1]) * fusion_1
        gate1 = torch.sigmoid(self.gate_params[1]).view(1, -1, 1, 1, 1)
        main_1 = main_1 + gate1 * fusion_1

        # Stage 2: 处理各路径
        main_2 = self.layer2(main_1)
        roi_2 = self.roi_layer2(roi_1)
        t1_2 = self.t1_layer2(t1_1)

        # Stage 2: 门控融合
        fusion_2 = self.amf_modules[2](main_2, roi_2, t1_2)
        # main_2 = main_2 + torch.sigmoid(self.gate_params[2]) * fusion_2
        gate2 = torch.sigmoid(self.gate_params[2]).view(1, -1, 1, 1, 1)
        main_2 = main_2 + gate2 * fusion_2

        # Stage 3: 处理各路径
        main_3 = self.layer3(main_2)
        roi_3 = self.roi_layer3(roi_2)
        t1_3 = self.t1_layer3(t1_2)

        # Stage 3: 门控融合
        fusion_3 = self.amf_modules[3](main_3, roi_3, t1_3)
        # main_3 = main_3 + torch.sigmoid(self.gate_params[3]) * fusion_3
        gate3 = torch.sigmoid(self.gate_params[3]).view(1, -1, 1, 1, 1)
        main_3 = main_3 + gate3 * fusion_3

        # Stage 4: 最终处理和分类
        out = self.layer4(main_3)
        out = self.dropblock(out)
        out = self.global_pool(out)
        out = out.flatten(1)
        out = self.classifier(out)

        return out


def create_pd_amfnet(**kwargs):
    """创建PD_AMFNet模型"""
    model = PD_AMFNet_Gated(
        layers=[3, 4, 6, 3],
        base_width=16,
        cardinality=8,
        num_classes=1,
        drop_prob=0.1
    )
    return model


