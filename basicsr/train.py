import datetime
import logging
import math
import time
import torch
from os import path as osp
import cv2
import numpy as np
import os
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters

def save_feature_as_cvpr(features_dict, save_path_prefix):
    """
    独立保存每个尺度的原始尺寸特征图。
    features_dict: {'res_orig': tensor, 'res_mid': tensor, 'res_low': tensor}
    """


    for key, feat in features_dict.items():
        # 1. 通道降维 [1, C, H, W] -> [H, W]
        # 使用 max 激活（Max Pooling 降维）能更好地突出 SR 任务中的结构细节
        heatmap, _ = torch.max(feat, dim=1)
        heatmap = heatmap.squeeze().detach().cpu().numpy()

        # 2. 归一化 (Min-Max)
        h_min, h_max = heatmap.min(), heatmap.max()
        heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)

        # 3. 伪彩色映射: COLORMAP_INFERNO 提供黑-紫-红-黄的平滑过渡
        color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)

        # 4. 独立保存原图大小，文件名区分尺度
        save_path = f"{save_path_prefix}_{key}.png"
        cv2.imwrite(save_path, color_map)


def save_feature_maps(model, val_loader, current_iter, opt):
    import os
    # 创建可视化目录
    save_dir = os.path.join(opt['path']['visualization'], f'features_iter_{current_iter}')
    os.makedirs(save_dir, exist_ok=True)

    # 兼容单卡或分布式模式
    net = model.net_g
    net.eval()

    count = 0
    with torch.no_grad():
        for val_data in val_loader:
            if count >= 40: break  # 只处理前40张

            # 推理一次以触发 MS_FMB 内部的特征记录
            input_img = val_data['lq'].to(model.device)
            _ = net(input_img)

            # 获取 body 中第一个 MS_FMB 模块的特征
            target_block = net.body[0]
            if hasattr(target_block, 'last_features'):
                features_dict = target_block.last_features

                # 构建文件名前缀
                img_path = val_data['lq_path'][0]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                save_prefix = os.path.join(save_dir, img_name)

                # 【关键】调用时不带 target_size 参数
                save_feature_as_cvpr(features_dict, save_prefix)

            count += 1
    # 切回训练模式
    net.train()

def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):

    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True

    resume_state = load_resume_state(opt)

    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))


    copy_opt_file(args.opt, opt['path']['experiments_root'])


    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    tb_logger = init_tb_loggers(opt)


    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    # --- 在这里插入自定义逻辑 ---
                    save_feature_maps(model, val_loader, current_iter, opt)
                    # --- 原有逻辑 ---
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    import sys
    import os.path as osp

    # 更加强劲的注入方式
    if len(sys.argv) == 1:
        # 尝试同时兼容单横线和双横线的情况
        sys.argv.extend(['-opt', r'D:\aaaasr\GACNet-Master-main\options\train\train_GCANet_x4.yml'])

    # 自动定位项目根目录 (通常是 train.py 的上两级)
    # 确保你的 options 文件夹在 root_path 下面
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))

    # 打印一下，方便你在控制台看到当前的运行路径，防止找不到 yml 文件
    print(f"Current root path: {root_path}")
    print(f"Arguments: {sys.argv}")

    train_pipeline(root_path)
# if __name__ == '__main__':
#     import sys
#     import os.path as osp
#     import torch
#     import time
#     from basicsr.utils.options import parse_options
#     from basicsr.models import build_model
#
#     # 尝试导入 thop
#     try:
#         from thop import profile
#     except ImportError:
#         print("Please install thop first: pip install thop")
#         sys.exit(1)
#
#     # 1. 注入配置文件路径
#     if len(sys.argv) == 1:
#         sys.argv.extend(['-opt', r'E:\zyl\crossSR\LKFMixer-master\options\train\train_LKFMixer_x4.yml'])
#
#     # 自动定位项目根目录
#     root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
#     opt_path = sys.argv[sys.argv.index('-opt') + 1]
#
#     print(f"Current root path: {root_path}")
#     print(f"Using config: {opt_path}")
#
#     # 2. 解析配置文件
#     opt, _ = parse_options(opt_path, is_train=False)
#     opt['root_path'] = root_path
#     opt['dist'] = False
#
#     # 3. 构建模型 (从 opt 中提取 network_g 定义并初始化)
#     print(f"Building model: {opt['network_g']['type']}...")
#     model = build_model(opt)
#     net = model.net_g.cuda()  # 提取真正的网络定义
#     net.eval()
#
#     # 4. 准备输入数据
#     # 保持 192x192 尺寸以便与 RGT 进行公平对比
#     input_size = (1, 3, 128, 128)
#     dummy_input = torch.randn(*input_size).cuda()
#
#     # 5. 计算 FLOPs 和参数量
#     print(f"--- Benchmarking with input size {input_size} ---")
#     flops, params = profile(net, inputs=(dummy_input,))
#
#     print("-" * 30)
#     print(f"LKFMixer Evaluation Results")
#     print(f"Total FLOPs: {flops / 1e9:.2f} G")
#     print(f"Total Params: {params / 1e6:.2f} M")
#     print("-" * 30)
#
#     # 6. 推理时间测试
#     iterations = 100  # 增加到100次以获得稳定的平均值
#     warmup_iters = 20 # 增加预热次数
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
#
#     with torch.no_grad():
#         # 预热阶段：让显卡进入高性能状态，加载缓存
#         for _ in range(warmup_iters):
#             _ = net(dummy_input)
#
#         if device.type == 'cuda':
#             # 使用 CUDA Event 进行精确计时
#             starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#             torch.cuda.synchronize()
#             starter.record()
#
#             for _ in range(iterations):
#                 _ = net(dummy_input)
#
#             ender.record()
#             torch.cuda.synchronize()  # 关键：等待所有 GPU 核完成工作
#             curr_time = starter.elapsed_time(ender)
#             avg_time = curr_time / iterations
#         else:
#             # CPU 计时逻辑
#             start_time = time.time()
#             for _ in range(iterations):
#                 _ = net(dummy_input)
#             avg_time = (time.time() - start_time) * 1000 / iterations
#
#     print(f"Average Inference Time: {avg_time:.2f} ms")
#     print(f"FPS: {1000 / avg_time:.2f}")
#     print("-" * 30)