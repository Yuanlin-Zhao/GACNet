import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights

class ADLK(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.k = kernel_size
        # 垂直与水平分解，减少参数量同时建立方向敏感性
        self.dw_h = nn.Conv2d(channels, channels, (kernel_size, 1),
                              padding=(kernel_size // 2, 0), groups=channels)
        self.dw_v = nn.Conv2d(channels, channels, (1, kernel_size),
                              padding=(0, kernel_size // 2), groups=channels)

        # 梯度感知分支：利用局部方差估计热梯度方向
        self.gradient_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # 模拟各向异性扩散：$\frac{\partial I}{\partial t} = div(c(x,y,t) \nabla I)$
        # 这里用 gradient_gate 模拟扩散系数 c
        grad = self.gradient_gate(x)
        diffused = self.dw_h(x) + self.dw_v(x)
        out = x + diffused * grad
        return self.proj(out)



class FAD(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.distill = nn.Conv2d(channels, channels // 2, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )
        # 频率补偿：全局上下文捕获
        self.global_context = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # 蒸馏出一部分低频纯净特征
        low_freq = self.distill(x)
        # 对剩余特征进行精炼
        refined = self.refine(x)
        # 全局热分布补偿
        context = self.fc(self.global_context(x))
        return refined + context, low_freq



class CAHM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 超网络：预测特征图的仿射变换参数
        self.hyper = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels * 2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        stats = self.gap(x).view(b, c)
        # 生成 scale (alpha) 和 shift (beta)
        params = self.hyper(stats).view(b, 2, c, 1, 1)
        alpha = torch.sigmoid(params[:, 0, :])
        beta = params[:, 1, :]
        return x * alpha + beta

class FMB(nn.Module):
    def __init__(self, channels, large_kernel, split_factor):
        super().__init__()
        self.split_channels = int(channels * split_factor)

        # 分支 1: 各向异性扩散大核（处理结构边缘）
        self.adlk = ADLK(self.split_channels, large_kernel)

        # 分支 2: 频率感知蒸馏（处理噪声与低频细节）
        self.fad = FAD(channels - self.split_channels)

        # 融合后的动态调制
        self.modulation = CAHM(channels)

        self.final_conv = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        x1, x2 = torch.split(x, (self.split_channels, x.shape[1] - self.split_channels), dim=1)

        # 分支计算
        x1 = self.adlk(x1)
        x2_refined, x2_distill = self.fad(x2)

        # 拼接并融合蒸馏特征（TPAMI常见的通道聚合策略）
        out = torch.cat([x1, x2_refined], dim=1)

        # 动态热特征校正
        out = self.modulation(out)

        out = self.act(self.final_conv(out))
        return out + identity



class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.upsample(x)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(b, c, h, w)


class MultiScaleRepConv(nn.Module):
    """多尺度重参数化，捕捉超分所需的不同频率细节"""

    def __init__(self, channels):
        super(MultiScaleRepConv, self).__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, 1, 1)
        # 引入空洞卷积增加感受野，匹配 Swin 的长程能力
        self.conv3x3_d2 = nn.Conv2d(channels, channels, 3, 1, 2, dilation=2)
        self.conv1x1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # 训练时融合多尺度特征
        return self.act(self.conv3x3(x) + self.conv3x3_d2(x) + self.conv1x1(x))


class X_MASP_Module(nn.Module):
    def __init__(self, channels, factor=4):
        super(X_MASP_Module, self).__init__()
        self.groups = factor
        group_ch = channels // self.groups

        # 1. 组间通讯
        self.shuffle = ChannelShuffle(factor)

        # 2. 坐标感知路径 (增强型：Avg + Max)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, group_ch // 4)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(group_ch, mip, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mip, group_ch, 1)
        )

        # 3. 细节增强
        self.detail_conv = MultiScaleRepConv(group_ch)

        # 4. 像素注意力
        self.pa = nn.Sequential(
            nn.Conv2d(group_ch, group_ch, 1),
            nn.Sigmoid()
        )

        # 5. LayerScale: 非常重要，保证初始化时是恒等映射
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, x):
        b, c, h, w = x.size()

        # 组间信息交换
        x_shuffled = self.shuffle(x)

        # 分组处理
        group_x = x_shuffled.reshape(b * self.groups, -1, h, w)
        n, c_g, _, _ = group_x.size()

        # --- 空间路径：融合平均值和最大值以捕捉边缘 ---
        h_feat = self.pool_h(group_x) + self.max_pool_h(group_x)
        w_feat = (self.pool_w(group_x) + self.max_pool_w(group_x)).permute(0, 1, 3, 2)

        y = torch.cat([h_feat, w_feat], dim=2)
        y = self.spatial_conv(y)

        x_h_sig, x_w_sig = torch.split(y, [h, w], dim=2)
        spatial_att = x_h_sig.sigmoid() * x_w_sig.permute(0, 1, 3, 2).sigmoid()

        # --- 特征提取 ---
        x_spatial = group_x * spatial_att
        x_details = self.detail_conv(group_x)

        # --- 融合 ---
        pixel_weight = self.pa(x_spatial + x_details)
        out = (group_x * pixel_weight) + group_x

        # 还原形状
        out = out.reshape(b, c, h, w)

        # 应用 LayerScale 残差
        return x + out * self.gamma.view(1, -1, 1, 1)


class MASP(nn.Module): #MASP
    def __init__(self, channels, factor=4):
        super().__init__()
        self.x_MASP = X_MASP_Module(channels, factor=factor)
        # 最后的 3x3 卷积用于平滑特征
        self.conv_end = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        # 模块内部已经包含了输入 x 的残差，这里直接返回
        return self.conv_end(self.x_MASP(x))

@ARCH_REGISTRY.register()
class LKFMixer(nn.Module):
    def __init__(self, in_channels, channels, out_channels, upscale, num_block, large_kernel, split_factor):
        super(LKFMixer, self).__init__()
        # 浅层特征提取
        self.conv_first = nn.Conv2d(in_channels, channels, 3, 1, 1)

        # 深层非线性映射：使用改进后的 FMB
        self.body = nn.ModuleList([
            FMB(channels, large_kernel, split_factor) for _ in range(num_block)
        ])
        self.conv_after_body = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)

        # 上采样
        self.upsampler = PixelShuffleDirect(scale=upscale, num_feat=channels, num_out_ch=out_channels)
        self.act = nn.GELU()
        self.m = MASP(channels)
    def forward(self, input):

        fea = self.conv_first(input)
        out = fea
        for block in self.body:
            out = block(out)
        out = self.m(out)
        out = self.act(self.conv_after_body(out))
        output = self.upsampler(out + fea)
        return output
from thop import profile

    # 测试 24 层深度的配置
model = LKFMixer(num_block=24, large_kernel=31, channels=64)
input_data = torch.randn(1, 3, 320, 180)
flops, params = profile(model, inputs=(input_data,))
print(f"\n[ALE-GMN TPAMI] GFLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")

# --- 测试与性能评估 ---
if __name__ == "__main__":
    from thop import profile

    # 参数配置完全遵循原始 LKFMixer 接口
    model = LKFMixer(
        in_channels=3,
        channels=64,
        out_channels=3,
        upscale=4,
        num_block=12,
        large_kernel=31,
        split_factor=0.25
    )

    input_data = torch.randn(1, 3, 320, 180)
    flops, params = profile(model, inputs=(input_data,))

    print("-" * 30)
    print(f"TPAMI-Style LKFMixer Evaluation")
    print(f"Total FLOPs: {flops / 1e9:.2f} G")
    print(f"Total Params: {params / 1e6:.2f} M")
    print("-" * 30)