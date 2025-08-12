# class EarlyStopping:
#     def __init__(self, patience=20, min_delta=0):
#         """
#         初始化早停类
#         :param patience: 允许验证损失不再改善的 epoch 数
#         :param min_delta: 验证损失的最小改善量
#         """
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0  # 记录验证损失没有改善的连续 epoch 数
#         self.best_loss = None  # 记录最佳的验证损失
#         self.early_stop = False  # 是否触发早停的标志
#
#     def __call__(self, val_loss):
#         """
#         每次验证后调用，更新早停状态
#         :param val_loss: 当前 epoch 的验证损失
#         """
#         if self.best_loss is None:
#             # 如果是第一次调用，直接记录当前验证损失
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.min_delta:
#             # 如果当前验证损失没有显著改善
#             self.counter += 1
#             if self.counter >= self.patience:
#                 # 如果连续 patience 个 epoch 没有改善，触发早停
#                 self.early_stop = True
#         else:
#             # 如果当前验证损失有显著改善，更新最佳损失并重置 counter
#             self.best_loss = val_loss
#             self.counter = 0

import numpy as np


class EarlyStopping:
    """早停策略，监控验证集的损失，防止过拟合"""

    def __init__(self, patience=20, min_delta=0, verbose=False):
        """
        参数:
            patience: 能够容忍的验证集损失不下降的轮数
            min_delta: 最小变化阈值，小于此值的变化视为没有改进
            verbose: 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0