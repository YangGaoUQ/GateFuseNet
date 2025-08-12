import random
import os
import numpy as np
import pandas as pd
import torch


def read_data_split(file_path):
    """读取数据划分文件并返回每一折的训练集和验证集文件列表"""
    folds = []
    print(os.getcwd())
    with open(file_path, 'r') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        if lines[idx].startswith("fold"):
            fold_idx = int(lines[idx].split()[1]) - 1

            train_files, val_files = [], []

            idx += 2  # Skip "train_files:" line

            # 读取训练集文件路径
            while not lines[idx].startswith("val_files:"):
                train_files.append(lines[idx].strip())
                idx += 1

            idx += 1  # Skip "val_files:" line

            # 读取验证集文件路径
            while idx < len(lines) and lines[idx].strip():
                val_files.append(lines[idx].strip())
                idx += 1

            folds.append((train_files, val_files))

        idx += 1
    return folds   #五个元组，每一个元组包含两个列表，第一个(维)列表是训练集文件路径，第二个(维)列表是验证集文件路径

def read_test_data(file_path):
    """
    从文件中读取数据路径和标签
    文件格式: QSM路径 T1路径 Seg路径 标签
    返回字符串列表，每个字符串是文件中的一行
    """
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # 去除换行符和多余空格
            if line:  # 确保行不为空
                samples.append(line)
    return samples
# def read_test_data(file_path):
#     """
#     从文件中读取数据路径和标签
#     文件格式: QSM路径 T1路径 Seg路径 标签
#     """
#     samples = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 4:
#                 qsm_path, t1_path, seg_path, label = parts
#                 samples.append((qsm_path, t1_path, seg_path, int(label)))
#     return samples


 # 加载训练好的模型权重
def load_best_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换到评估模式
    return model


def save_results_to_excel(results, file_path):
    """
    将每个折的结果和平均结果保存到Excel文件
    :param results: 包含每个折的结果和平均结果的字典
    :param file_path: 保存结果的Excel文件路径
    """
    # 将每个折的结果转化为 DataFrame
    fold_results = []
    for fold, metrics in results["folds"].items():
        metrics["fold"] = fold
        fold_results.append(metrics)

    # 将平均结果转化为 DataFrame
    avg_results = results["average"]
    avg_results["fold"] = "average"
    fold_results.append(avg_results)

    # 将结果存储为 DataFrame
    df = pd.DataFrame(fold_results)

    # 保存到Excel文件
    df.to_excel(file_path, index=False)


def set_random_seed(seed=42):
    # 设置Python的哈希种子
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 设置随机数生成器的种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 设置GPU的种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保cuDNN使用确定性算法
    torch.backends.cudnn.deterministic = True

    # 禁用cuDNN的自动调优（确保复现性）
    torch.backends.cudnn.benchmark = False


if __name__=='__main__':
    file_path = '../train_val_splits2.txt'
    data_splits = read_data_split(file_path)
    # print(data_splits)
    print(len(data_splits))
    #打印data_splits的类型和形状大小
    print(data_splits[0])
    print(data_splits[4])
    print(len(data_splits[0]))
    print(len(data_splits[0][0]))
    print(data_splits[0][0][0])


    # train