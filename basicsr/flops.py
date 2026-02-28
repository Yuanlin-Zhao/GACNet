import sys
import os.path as osp
import torch
import time
from thop import profile
from basicsr.utils.options import parse_options
from basicsr.models import build_model


def benchmark_pipeline(root_path):
    # 1. 解析配置文件
    opt_path = sys.argv[sys.argv.index('-opt') + 1]
    opt, _ = parse_options(opt_path, is_train=False)
    opt['root_path'] = root_path
    opt['dist'] = False  # 非分布式模式

    # 2. 构建模型 (BasicSR 包装器)
    print(f"--> 正在构建模型: {opt['network_g']['type']}...")
    model = build_model(opt)

    # 3. 提取纯网络对象并设为评估模式
    net = model.net_g.cuda()
    net.eval()


    input_size = (1, 3, 160, 128)
    dummy_input = torch.randn(*input_size).cuda()

    print(f"\n" + "=" * 40)
    print(f"性能测试报告 (输入尺寸: {input_size})")
    print("=" * 40)

    # 5. 计算 FLOPs 和 Params
    # 使用 thop 进行分析
    flops, params = profile(net, inputs=(dummy_input,), verbose=False)

    print(f"总参数量 (Params): {params / 1e6:.2f} M")
    print(f"总计算量 (FLOPs): {flops / 1e9:.2f} G")
    print("-" * 40)

    # 6. 推理时间测试 (Latency & FPS)
    iterations = 100
    warmup_iters = 20

    print(f"开始推理速度测试 (迭代 {iterations} 次)...")

    with torch.no_grad():
        # GPU 预热 (让显卡脱离节能模式，加载算子缓存)
        for _ in range(warmup_iters):
            _ = net(dummy_input)

        # 高精度计时
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        starter.record()

        for _ in range(iterations):
            _ = net(dummy_input)

        ender.record()
        torch.cuda.synchronize()  # 等待 GPU 完成所有任务

        total_time_ms = starter.elapsed_time(ender)
        avg_time_ms = total_time_ms / iterations

    print("-" * 40)
    print(f"平均推理延迟: {avg_time_ms:.2f} ms")
    print(f"每秒帧数 (FPS): {1000 / avg_time_ms:.2f}")
    print("=" * 40)


if __name__ == '__main__':
    # 注入配置文件路径
    if len(sys.argv) == 1:
        # 指向你的 LKFMixer 配置文件
        sys.argv.extend(['-opt', r'E:\zyl\crossSR\LKFMixer-master\options\train\train_LKFMixer_x4.yml'])

    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))

    print(f"Current root path: {root_path}")

    # 执行性能测试而不是执行训练
    benchmark_pipeline(root_path)