# 分类指标计算
# def compute_classification_metrics(true_labels, predictions):
#     accuracy = accuracy_score(true_labels, predictions)
#     f1 = f1_score(true_labels, predictions)
#     precision = precision_score(true_labels, predictions)
#     recall = recall_score(true_labels, predictions)
#
#     return accuracy, f1, precision, recall
import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import seaborn as sns

from tensorflow.python.training.training_util import global_step

#
# def evaluate_metrics(labels, preds, probs):
#     """
#     计算并返回一系列评估指标：准确率、精确度、召回率、F1分数、AUC
#     :param labels: 真实标签
#     :param preds: 预测标签
#     :param probs: 预测的概率
#     :return: 一个包含所有指标的字典
#     """
#     accuracy = accuracy_score(labels, preds)
#     precision = precision_score(labels, preds)
#     recall = recall_score(labels, preds)
#     f1 = f1_score(labels, preds)
#     auc = roc_auc_score(labels, probs)
#
#     return {
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1_score": f1,
#         "auc": auc
#     }
#
# def plot_curve(metric, train_values, val_values, fold_idx, save_dir, x_values=None):
#     """
#     绘制训练和验证曲线
#     :param metric: 指标名称（如 'Loss', 'AUC'）
#     :param train_values: 训练集指标值列表
#     :param val_values: 验证集指标值列表
#     :param fold_idx: 当前 fold 的索引
#     :param save_dir: 图片保存路径
#     :param x_values: 横坐标的值（可选，默认为 range(1, len(train_values) + 1)）
#     """
#     # 确保保存路径存在
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 如果未提供 x_values，则默认为 range(1, len(train_values) + 1)
#     if x_values is None:
#         x_values = range(1, len(train_values) + 1)
#
#     plt.figure()
#     # 绘制训练和验证曲线
#     plt.plot(x_values, train_values, label=f'Train {metric}')
#     plt.plot(x_values, val_values, label=f'Val {metric}')
#     plt.title(f'{metric} Curve Fold {fold_idx}')
#     plt.xlabel('Epoch')
#     plt.ylabel(metric)
#     plt.legend()
#     # 保存图像
#     plt.savefig(os.path.join(save_dir, f'{metric.lower()}_curve_fold_{fold_idx}.png'))
#     plt.close()
#
#
# def plot_roc_curve(labels, probs, fold_idx=None,save_path=None):
#     fpr, tpr, _ = roc_curve(labels, probs)
#     roc_auc = auc(fpr, tpr)
#
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve (Fold {fold_idx})' if fold_idx is not None else 'ROC Curve')
#     plt.legend(loc="lower right")
#
#     if save_path is not None:
#         # 确保保存路径存在
#         os.makedirs(save_path, exist_ok=True)
#         # 生成文件名
#         if fold_idx is not None:
#             save_file = os.path.join(save_path, f'ROC_Curve_Fold_{fold_idx}.png')
#         else:
#             save_file = os.path.join(save_path, 'ROC_Curve.png')
#         # 保存图像
#         plt.savefig(save_file)
#         print(f"ROC curve saved at {save_file}")
#
#     plt.show()
#
#
# def plot_confusion_matrix(cm, class_names, writer=None, epoch=None, tag='Confusion_Matrix', save_path=None,
#                           use_sns=True):
#     """
#     通用混淆矩阵绘制函数，支持TensorBoard记录和图片保存
#     :param cm: 混淆矩阵
#     :param class_names: 类别名称
#     :param writer: TensorBoard writer
#     :param epoch: 当前epoch
#     :param tag: TensorBoard中的图像名称
#     :param save_path: 图片保存路径 (可选)
#     :param use_sns: 是否使用seaborn绘制混淆矩阵 (默认为True)
#     """
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     if use_sns:
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True)
#     else:
#         ax.imshow(cm, cmap='Blues')
#         ax.set_xticks(np.arange(len(class_names)))
#         ax.set_yticks(np.arange(len(class_names)))
#         ax.set_xticklabels(class_names)
#         ax.set_yticklabels(class_names)
#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#         for i in range(len(class_names)):
#             for j in range(len(class_names)):
#                 ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
#
#     plt.title(tag.replace('_', ' '))
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.tight_layout()
#
#     # 如果需要记录到TensorBoard
#     if writer is not None and epoch is not None:
#         writer.add_figure(tag, fig, epoch)
#
#     # 保存本地图片
#     if save_path is not None:
#         # 确保保存路径存在
#         os.makedirs(save_path, exist_ok=True)
#         # 拼接完整路径
#         save_file = os.path.join(save_path, f'{tag}_epoch_{epoch}.png')
#         plt.savefig(save_file)
#         print(f"Confusion Matrix saved at {save_file}")
#
#     plt.close(fig)
#
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    roc_curve, auc
)
import seaborn as sns


def evaluate_metrics(labels, preds, probs):
    """
    计算并返回精选的评估指标，专门针对医疗分类任务优化

    Args:
        labels: 真实标签
        preds: 预测标签
        probs: 预测的概率

    Returns:
        一个包含所有指标的字典
    """
    # 确保输入是numpy数组并展平
    labels = np.array(labels).flatten()
    preds = np.array(preds).flatten()

    # 检查是否有足够的样本
    if len(labels) == 0:
        return {}

    # 基本分类指标
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1_score": f1_score(labels, preds, zero_division=0),
    }

    # 混淆矩阵相关指标
    if len(np.unique(labels)) > 1:
        try:
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

            # 敏感性 (Sensitivity) = 召回率 (Recall)
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0

            # 特异性 (Specificity) = 真阴性率
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

            # 阳性预测值 (PPV) = 精确率 (Precision)
            metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0

            # 阴性预测值 (NPV)
            metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0

            # 约登指数 (Youden Index) = 敏感性 + 特异性 - 1
            metrics["youden_index"] = metrics["sensitivity"] + metrics["specificity"] - 1

        except ValueError:
            # 如果混淆矩阵无法计算，设置默认值
            metrics.update({
                "sensitivity": np.nan,
                "specificity": np.nan,
                "ppv": np.nan,
                "npv": np.nan,
                "youden_index": np.nan
            })
    else:
        # 只有一个类别时的处理
        metrics.update({
            "sensitivity": np.nan,
            "specificity": np.nan,
            "ppv": np.nan,
            "npv": np.nan,
            "youden_index": np.nan
        })

    # 基于概率的指标
    if probs is not None and len(np.unique(labels)) > 1:
        try:
            # AUC-ROC
            metrics["auc_roc"] = roc_auc_score(labels, probs)

            # AUC-PR (平均精确率)
            metrics["auc_pr"] = average_precision_score(labels, probs)

        except Exception as e:
            warnings.warn(f"概率相关指标计算失败: {e}")
            metrics.update({
                "auc_roc": 0.5,
                "auc_pr": 0.0
            })
    else:
        # 概率为空或只有一个类别时的处理
        metrics.update({
            "auc_roc": 0.5,
            "auc_pr": 0.0
        })

    return metrics

# def evaluate_metrics(labels, preds, probs):
#     """
#     计算并返回一系列评估指标：准确率、精确度、召回率、F1分数、AUC、特异性
#
#     Args:
#         labels: 真实标签
#         preds: 预测标签
#         probs: 预测的概率
#
#     Returns:
#         一个包含所有指标的字典
#     """
#     # 确保输入是numpy数组并展平
#     labels = np.array(labels).flatten()
#     preds = np.array(preds).flatten()
#
#     # 基本指标
#     metrics = {
#         "accuracy": accuracy_score(labels, preds),
#         "precision": precision_score(labels, preds, zero_division=0),
#         "recall": recall_score(labels, preds, zero_division=0),
#         "f1_score": f1_score(labels, preds, zero_division=0)
#     }
#
#     # 计算特异性
#     if len(np.unique(labels)) > 1:  # 确保有正负两类样本
#         tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
#         metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
#     else:
#         metrics["specificity"] = np.nan
#
#     # 计算AUC
#     if probs is not None and len(np.unique(labels)) > 1:
#         metrics["auc"] = roc_auc_score(labels, probs)
#     else:
#         metrics["auc"] = 0.5  # 只有一个类别时AUC无意义
#
#     return metrics


def plot_curve(metric, train_values, val_values, fold_idx, save_dir, x_values=None):
    """
    绘制训练和验证曲线

    Args:
        metric: 指标名称（如 'Loss', 'AUC'）
        train_values: 训练集指标值列表
        val_values: 验证集指标值列表
        fold_idx: 当前 fold 的索引
        save_dir: 图片保存路径
        x_values: 横坐标的值（可选，默认为 range(1, len(train_values) + 1)）
    """
    # 确保保存路径存在
    os.makedirs(save_dir, exist_ok=True)

    # 如果未提供 x_values，则默认为 range(1, len(train_values) + 1)
    if x_values is None:
        x_values = range(1, len(train_values) + 1)

    plt.figure()
    # 绘制训练和验证曲线
    plt.plot(x_values, train_values, label=f'Train {metric}')
    plt.plot(x_values, val_values, label=f'Val {metric}')
    plt.title(f'{metric} Curve Fold {fold_idx}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)  # 添加网格

    # 保存图像
    plt.savefig(os.path.join(save_dir, f'{metric.lower()}_curve_fold_{fold_idx}.png'))
    plt.close()


def plot_roc_curve(labels, probs, fold_idx=None, save_path=None, title=None, show=False):
    """
    绘制ROC曲线

    Args:
        labels: 真实标签
        probs: 预测的概率
        fold_idx: 折叠索引（可选）
        save_path: 保存路径（可选）
        title: 图表标题（可选）
        show: 是否显示图形
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # 设置标题
    if title:
        plt.title(title)
    else:
        plt.title(f'ROC Curve (Fold {fold_idx})' if fold_idx is not None else 'ROC Curve')

    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path is not None:
        # 确保保存路径存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        # 保存图像
        plt.savefig(save_path)
        print(f"ROC curve saved at {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return roc_auc


def plot_confusion_matrix(cm, class_names, writer=None, epoch=None, tag='Confusion_Matrix', save_path=None,
                          use_sns=True):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        writer: TensorBoard writer
        epoch: 当前epoch
        tag: TensorBoard中的图像名称
        save_path: 图片保存路径 (可选)
        use_sns: 是否使用seaborn绘制混淆矩阵 (默认为True)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if use_sns:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True)
    else:
        ax.imshow(cm, cmap='Blues')
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.title(tag.replace('_', ' '))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # 如果需要记录到TensorBoard
    if writer is not None and epoch is not None:
        writer.add_figure(tag, fig, epoch)

    # 保存本地图片
    if save_path is not None:
        # 确保保存路径存在
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(save_dir, exist_ok=True)

        # 保存图像
        plt.savefig(save_path)
        print(f"Confusion Matrix saved at {save_path}")

    plt.close(fig)

    return fig


def plot_combined_roc_curves(results_dict, save_path=None, show=False):
    """
    绘制多个模型/折叠的ROC曲线对比

    Args:
        results_dict: 字典，键为模型/折叠名称，值为包含labels和probs的元组
        save_path: 保存路径（可选）
        show: 是否显示图形
    """
    plt.figure(figsize=(10, 8))

    colors = ['darkorange', 'limegreen', 'cornflowerblue', 'purple', 'crimson',
              'gold', 'teal', 'darkviolet', 'sienna', 'darkslategray']

    for i, (name, (labels, probs)) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path is not None:
        # 确保保存路径存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        # 保存图像
        plt.savefig(save_path)
        print(f"Combined ROC curves saved at {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
