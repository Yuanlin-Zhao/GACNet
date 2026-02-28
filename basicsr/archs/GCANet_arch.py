
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile  # 计算参数量和运算量
from basicsr.archs.arch_util import default_init_weights
class PixelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # (B, C, H, W) -> (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()

class DLK(nn.Module):
    def __init__(self, channels, kernel_size=31, split_factor=0.25):
        super().__init__()
        self.split_channels = int(channels * split_factor)
        self.remain_channels = channels - self.split_channels
        print(kernel_size)
        # 确保 k 是奇数
        k = kernel_size // 1
        if k % 2 == 0:
            k = k + 1  # 偶数变奇数
        self.local_dw = nn.Conv2d(channels, channels, 5, padding=2)
        self.global_dw = nn.Sequential(
            nn.Conv2d(channels, channels, (k, 1),
                      padding=(k // 2, 0), groups=self.split_channels),
            nn.Conv2d(channels, channels, (1, k),
                      padding=(0, k // 2), groups=self.split_channels),
        )
        self.project = nn.Conv2d(channels * 2, channels, 1)
        self.act = nn.GELU()
    def forward(self, x):
        x1 = x.clone()
        x2 = x.clone()
        x1 = self.local_dw(x1) + self.global_dw(x1)
        out = torch.cat([x1, x2], dim=1)
        return self.act(self.project(out))


class TDA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 这里的 channels 必须是完整的通道数 (例如 64)
        self.context_weight = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),  # 注意这里：输入应为 channels // 4
            nn.Sigmoid()
        )
        self.hp_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x):
        # 确保此时 x 的 shape 是 [B, 64, H, W]
        b, c, h, w = x.shape
        weight = self.context_weight(self.avg_pool(x))

        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - low_freq
        high_freq = self.hp_conv(high_freq)

        return x * weight + high_freq

# class HAFM(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # 这里的 channels 必须是完整的通道数 (例如 64)
#         self.context_weight = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, 1),
#             nn.GELU(),
#             nn.Conv2d(channels // 4, channels, 1),  # 注意这里：输入应为 channels // 4
#             nn.Sigmoid()
#         )
#         self.hp_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
#
#     def forward(self, x):
#         # 确保此时 x 的 shape 是 [B, 64, H, W]
#         b, c, h, w = x.shape
#         weight = self.context_weight(self.avg_pool(x))
#
#         low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         high_freq = x - low_freq
#         high_freq = self.hp_conv(high_freq)
#
#         return x * weight + high_freq


class TISB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.branch2 = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        g = self.gate(torch.cat([x1, x2], dim=1))
        return x1 * g + x2 * (1 - g)
# class FSB(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.branch1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
#         self.branch2 = nn.Conv2d(channels, channels, 1)
#         self.gate = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         g = self.gate(torch.cat([x1, x2], dim=1))
#         return x1 * g + x2 * (1 - g)

#Trifurcated Information Stream
class TIS(nn.Module):
    def __init__(self, channels, large_kernel, split_factor):
        super().__init__()

        # --- 1. 主尺度分支 (Original Scale: 1x) ---
        self.dlk_orig = DLK(channels, large_kernel, split_factor)
        self.hafm_orig = TDA(channels)
        self.fsb_orig = TISB(channels)

        # --- 2. 中尺度分支 (Medium Scale: 1/2 size) ---
        self.dlk_mid = DLK(channels, large_kernel, split_factor)
        self.hafm_mid = TDA(channels)
        self.fsb_mid = TISB(channels)
        self.down_mid = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

        # --- 3. 小尺度分支 (Small Scale: 1/4 size) ---
        self.dlk_low = DLK(channels, large_kernel, split_factor)
        self.hafm_low = TDA(channels)
        self.fsb_low = TISB(channels)
        # 1/4 采样采用两次 stride=2 的卷积，能获得更好的抗混叠特征
        self.down_low = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        )

        # --- 4. 融合与归一化 ---
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # 三分支融合：channels * 3 -> channels
        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.norm = PixelNorm(channels)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        # 分支 1: 原始尺度计算
        res_orig = self.fsb_orig(self.hafm_orig(self.dlk_orig(x)))

        # 分支 2: 1/2 下采样计算
        x_mid = self.down_mid(x)
        res_mid = self.fsb_mid(self.hafm_mid(self.dlk_mid(x_mid)))

        res_mid_up = F.interpolate(res_mid, size=(H, W), mode='bilinear', align_corners=False)

        # 分支 3: 1/4 下采样计算 (新增层)
        x_low = self.down_low(x)
        res_low = self.fsb_low(self.hafm_low(self.dlk_low(x_low)))

        res_low_up = F.interpolate(res_low, size=(H, W), mode='bilinear', align_corners=False)

        # --- 多尺度融合 (Multi-scale Fusion) ---
        # 沿通道维度拼接: [1x, 1/2x_up, 1/4x_up]
        combined = torch.cat([res_orig, res_mid_up, res_low_up], dim=1)
        out = self.fusion(combined)

        # # --- 新增：仅在验证模式下保存特征引用，用于可视化 ---
        if not self.training:
            self.last_features = {
                'res_orig': res_orig.detach().cpu(),
                'res_mid': res_mid.detach().cpu(),
                'res_low': res_low.detach().cpu()
            }

        return identity + self.norm(out) * self.res_scale

#Granularity-Aligned Constrained Network
@ARCH_REGISTRY.register()
class GACNet(nn.Module):
    def __init__(self, in_channels=3, channels=56, out_channels=3, upscale=4, num_block=10, large_kernel=31,
                 split_factor=0.25):
        super().__init__()
        # 1. 浅层特征提取
        self.conv_first = nn.Conv2d(in_channels, channels, 3, padding=1)

        # 2. 深层特征提取 (堆叠 TIS)
        self.body = nn.ModuleList([
            TIS(channels, large_kernel, split_factor) for _ in range(num_block)
        ])

        # 3. 瓶颈层与多尺度聚合
        self.conv_after_body = nn.Conv2d(channels, channels, 3, padding=1)

        # 4. 高质量上采样
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, out_channels * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale)
        )
    def forward(self, x):
        # 输入：(B, 3, H, W)
        feat_initial = self.conv_first(x)
        res = feat_initial
        for block in self.body:
            res = block(res)
        res = self.conv_after_body(res)
        # 全局残差连接
        out = self.upsample(res + feat_initial)
        return out

# from thop import profile
#
#     # 参数配置完全遵循原始 LKFMixer 接口
# model = LKFMixer(
#         in_channels=3,
#         channels=64,
#         out_channels=3,
#         upscale=4,
#         num_block=12,
#         large_kernel=31,
#         split_factor=0.25
#     )
#
# input_data = torch.randn(1, 3, 64, 64)
# flops, params = profile(model, inputs=(input_data,))
#
# print("-" * 30)
# print(f"TPAMI-Style LKFMixer Evaluation")
# print(f"Total FLOPs: {flops / 1e9:.2f} G")
# print(f"Total Params: {params / 1e6:.2f} M")
# print("-" * 30)
# # 4. 测试推理时间 (Inference Time)
# print("开始测试推理速度...")
# iterations = 10
# warmup_iters = 3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")
# # 预热阶段：排除初始化延迟
# with torch.no_grad():
#     for _ in range(warmup_iters):
#         _ = model(input_data)
#
#     # 同步并开始计时
#     if device.type == 'cuda':
#         starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#         starter.record()
#
#         for _ in range(iterations):
#             _ = model(input_data)
#
#         ender.record()
#         torch.cuda.synchronize()  # 等待 GPU 计算完成
#         curr_time = starter.elapsed_time(ender)  # 单位是毫秒 (ms)
#         avg_time = curr_time / iterations
#     else:
#         # CPU 计时逻辑
#         start_time = time.time()
#         for _ in range(iterations):
#             _ = model(input_data)
#         avg_time = (time.time() - start_time) * 1000 / iterations  # 转换为 ms
#
# print(f"Average Inference Time: {avg_time:.2f} ms")
# print(f"FPS: {1000 / avg_time:.2f}")
# print("-" * 30)

# --- 测试与性能评估 ---
# if __name__ == "__main__":
#     from thop import profile
#
#     # 参数配置完全遵循原始 LKFMixer 接口
#     model = LKFMixer(
#         in_channels=3,
#         channels=64,
#         out_channels=3,
#         upscale=4,
#         num_block=12,
#         large_kernel=31,
#         split_factor=0.25
#     )
#
#     input_data = torch.randn(1, 3, 320, 180)
#     flops, params = profile(model, inputs=(input_data,))
#
#     print("-" * 30)
#     print(f"TPAMI-Style LKFMixer Evaluation")
#     print(f"Total FLOPs: {flops / 1e9:.2f} G")
#     print(f"Total Params: {params / 1e6:.2f} M")
#     print("-" * 30)