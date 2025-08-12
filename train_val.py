import math
import multiprocessing

# from dataset.MultiModal_Dataset3D_v1 import Patch_Dataset3D

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')


import argparse
import shutil
import warnings
import os

from dataset.DualDataset2 import DualDataset2
from dataset.TripleDataset2 import Triple_Dataset
# from dataset.TripleDataset2 import Triple_Dataset

from models.DenseNet_3D import densenet121_3d
from models.SE_ResNeXt import seresnext50
from models.ResNeXt_3D import resnext50_32x4d_3d as resnext50
from models.AG_SE_ResNeXt50_3_modes import AG_SE_ResNeXt50_model_3
from dataset.DualDataset import Dual_Dataset
from models.AG_SE_ResNeXt50 import AG_SE_ResNeXt50_model
from utils.early_stopping import EarlyStopping
from models.Vgg import vgg16 as vgg
# # 忽略包含 "pixdim[0]" 的日志消息
# class CustomFilter(logging.Filter):
#     def filter(self, record):
#         return not re.search(r"pixdim\[0\]", record.getMessage())
#
# logging.getLogger("py.warnings").addFilter(CustomFilter())

# os.chdir("/home/sunwenwen/JR/PD_classfication")#
print("CWD: ", os.getcwd())
warnings.filterwarnings("ignore", message=r"pixdim.*")
print("warnings: pixdim[0] (qfac) should be 1 (default) or -1; setting qfac to 1")
warnings.filterwarnings('ignore', category=UserWarning, module='nibabel')

import numpy as np
import torch
import torch.optim as optim
import logging
import time
from matplotlib import pyplot as plt
# from tensorboardX import writer
from torch.utils.data import DataLoader
from torch import nn
import os
import torchio as tio
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

# from dataset.SingleDataset import Single_Dataset3D
from dataset.newDataset import Single_Dataset3D
from utils.my_utils import read_data_split, set_random_seed
from utils.metrics import evaluate_metrics, plot_confusion_matrix, plot_curve, plot_roc_curve
from models.ResNet50_3D import resnet50 as resnet

from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
from utils.losses import FocalLoss
# from models.My_Model  import create_pd_moderateqfusionnet as mymodel
# from models.My_Model2 import create_improved_pd_fusionnet as mymodel
logging.getLogger('nibabel').setLevel(logging.ERROR)
# from models.PD_AMFNet import PD_AMFNet_model as mymodel
#from models.PD_AMF_c2 import create_pd_amfnet as mymodel
# from models.Improved_model import OIAM_ResNeXt50_model as mymodel
# from models.AG_SE_ResNeXt50_3 import AG_SE_ResNeXt50_model_3 as mymodel
from models.PD_AMF_C2_2 import create_pd_amfnet_two_branch as mymodel
from matplotlib import cm
# file_path1 = os.path.join('/home/sunwenwen/JR/PD_classfication/', 'train_val_splits2.txt') #
# file_path2 = os.path.join('/home/sunwenwen/JR/PD_classfication/', 'Output_ResNet(QSM_Seg)') #
file_path1=os.path.join(os.getcwd(), 'train_val_splits.txt')
file_path2=os.path.join(os.getcwd(),'Output/Output_GA')
print(file_path1)
print(file_path2)

logging.captureWarnings(False)  # 关闭日志对警告的捕获
def parse_args():
    # 使用 argparse 来解析命令行参数
    parser = argparse.ArgumentParser(description="3D Model Training Script")
    # 添加参数
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate 1e-4')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--num_folds', type=int, default=5, help='Total number of folds for cross-validation')
    # parser.add_argument('--data_split_file', type=str, default='./train_val_splits2.txt', help='Path to data split txt file')
    # parser.add_argument('--save_dir',type=str,default='Output_SE_ResNext(Seg)', help='Directory where the model or data will be saved.')# 设置默认保存路径
    parser.add_argument('--modes_num', type=int, default=3,help='Number of modalities (1 for single modality, 2 for dual modalities)')
    parser.add_argument('--data_split_file', type=str, default=file_path1,help='Path to data split txt file')
    parser.add_argument('--save_dir', type=str, default=file_path2 ,help='Directory where the model or data will be saved.')
    # parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), 'PD_classfication/Output_SE_ResNext(Seg)'))
    args = parser.parse_args()

    return args


def cosine_annealing_with_warmup(optimizer, warmup_epochs, total_epochs):
    """构建带预热的余弦退火学习率调度器"""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1. + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_idx, writer=None, modes=1):
    print(f'Train device:{device}')
    model.train()
    running_loss = 0.0  # 累积损失
    total_samples = 0  # 样本总数
    all_labels = []  # 存储所有真实标签
    all_preds = []  # 存储所有模型预测值
    all_probs = []

    for data in train_loader:
        # print(modes)
        if modes == 1:  # 单模态输入：(inputs, labels)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1).float()  # 标签形状变为 [batch_size, 1]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)

        elif modes == 2:  # 双模态输入：((img_data1, img_data2), labels)
            # 假设数据格式为 ( (tensor1, tensor2), labels )
            inputs, labels = data
            img_data1, img_data2 = inputs
            img_data1 = img_data1.to(device)
            img_data2 = img_data2.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(img_data1, img_data2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = img_data1.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        elif modes == 3:  # 三模态输入：((data1, data2, data3), labels)
            # 假设数据格式为 ( (tensor1, tensor2, tensor3), labels )
            inputs, labels = data
            data1, data2, data3 = inputs
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(data1, data2, data3)  # 假设模型可以接受三个输入
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = data1.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        else:
            raise ValueError("数据格式不支持：modes 必须是 1 (单模态), 2 (双模态) 或 3 (三模态)")
        probs = torch.sigmoid(outputs).squeeze(1)
        preds = (probs > 0.5).int()  # 预测为类别1的概率大于0.5时预测为类别1
        # print(f"outputs shape: {outputs.shape}")
        # print(f"probs shape: {probs.shape}")

        # 保存标签和预测结果
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())  # 保存概率值，用于计算AUC

    avg_loss = running_loss / total_samples  # 即epoch_avg_loss
    metrics = evaluate_metrics(all_labels, all_preds, all_probs)

    # writer.add_scalar('Train/Loss', avg_loss, epoch_idx)
    # writer.add_scalar('Train/Accuracy', metrics['accuracy'], epoch_idx)
    # writer.add_scalar('Train/Precision', metrics['precision'], epoch_idx)
    # writer.add_scalar('Train/Recall', metrics['recall'], epoch_idx)
    # writer.add_scalar('Train/F1', metrics['f1_score'], epoch_idx)
    # writer.add_scalar('Train/AUC', metrics['auc'], epoch_idx)

    return avg_loss, metrics


def val_one_epoch(model, val_loader, criterion, device, epoch_idx, class_names=None, writer=None, save_Path=None,
                  modes=1):
    if class_names is None:
        class_names = ['HC', 'PD']
    if save_Path:
        save_Path = os.path.join(save_Path, 'Confusion_Matrix_pic')

    model.eval()
    val_loss = 0.0
    total_samples = 0
    val_all_labels = []
    val_all_preds = []
    val_all_probs = []

    with torch.no_grad():
        for data in val_loader:
            if modes == 1:  # 单模态输入：(inputs, labels)
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1).float()
                outputs = model(inputs)

            elif modes == 2:  # 双模态输入：((img_data1, img_data2), labels)
                inputs, labels = data
                img_data1, img_data2 = inputs
                img_data1 = img_data1.to(device)
                img_data2 = img_data2.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1).float()
                outputs = model(img_data1, img_data2)

            elif modes == 3:  # 三模态输入：((data1, data2, data3), labels)
                inputs, labels = data
                data1, data2, data3 = inputs
                data1 = data1.to(device)
                data2 = data2.to(device)
                data3 = data3.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1).float()
                outputs = model(data1, data2, data3)

            else:
                raise ValueError("数据格式不支持：modes 必须是 1 (单模态), 2 (双模态) 或 3 (三模态)")

            # 计算损失
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size

            # 计算概率和预测
            probs = torch.sigmoid(outputs).squeeze(1)  # [batch_size, 1] -> [batch_size]
            preds = (probs > 0.5).int()  # 阈值0.5分类

            # 保存当前批次的标签和预测
            val_all_labels.extend(labels.cpu().numpy())
            val_all_preds.extend(preds.cpu().numpy())
            val_all_probs.extend(probs.detach().cpu().numpy())
    avg_loss = val_loss / total_samples

    # 把 list 转成 ndarray，保持数据对齐
    val_all_labels = np.array(val_all_labels)
    val_all_preds = np.array(val_all_preds)
    val_all_probs = np.array(val_all_probs)

    # 指标计算
    metrics = evaluate_metrics(val_all_labels, val_all_preds, val_all_probs)
    # plot_roc_curve(val_all_labels,val_all_probs)

    # 计算混淆矩阵
    cm = confusion_matrix(val_all_labels, val_all_preds)
    # plot_confusion_matrix(cm, class_names, writer, epoch_idx, tag='Val_Confusion_Matrix',save_path=save_Path)
    print(f"Confusion Matrix:\n{cm}")

    if writer:
        # plot_confusion_matrix(cm, class_names, writer, epoch_idx, tag='Val_Confusion_Matrix', save_path=save_Path)

         # 记录Loss & Metrics到TensorBoard
        writer.add_scalar('Val/Loss', avg_loss, epoch_idx)
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f'Val/{metric_name}', metric_value, epoch_idx)

    return avg_loss, metrics, cm


# 定义 3D 数据增强的变换操作
transform = tio.Compose([
    # 随机旋转图像，旋转角度范围为 -15 到 15 度,保持纵横比一致 以50%的概率进行旋转
    tio.RandomAffine(degrees=5, translation=(2, 2, 2),scales=(0.9, 1.1),isotropic=True, p=0.2),
    # 随机裁剪（在指定区域内裁剪图像）# 裁剪到128x128x128的大小# 以50%的概率裁剪
    # tio.RandomCrop(subject_size=(128, 128, 128), p=0.5),
    # 随机改变亮度，调整亮度的范围# 强度范围# 以20%的概率进行亮度调整
    tio.RandomBiasField(coefficients=0.3, p=0.1),
    # 随机进行仿射变换，变换范围为一个小区间# X, Y, Z方向的平移范围 # 缩放比例
    # 随机噪声，添加噪声# 噪声标准差
    tio.RandomNoise(mean=0, std=0.02, p=0.1),
    # 随机镜像翻转# 如果有多个键值，则对对应的键进行翻转# 在左右(LR), 前后(AP), 上下(SI)三个方向上随机翻转
    # tio.RandomFlip(keys=["image"], axes=("LR", "AP", "SI"), p=0.3),
    #tio.RandomFlip(keys=["image"], axes=("LR", "AP", "SI"), p=0.2),
])

def main():

    set_random_seed(42)  # 设置固定的随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 解析命令行参数
    args = parse_args()

    # 确保保存路径存在，不存在则创建
    save_dir = args.save_dir
    # 1. 如果路径不存在，则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"路径 {save_dir} 不存在，已创建！")
    else:
        print(f"路径 {save_dir} 已存在，跳过删除操作。")

    # 2. 创建子目录（如果不存在）
    confusion_matrix_dir = os.path.join(save_dir, 'Confusion_Matrix_pic')
    if not os.path.exists(confusion_matrix_dir):
        os.makedirs(confusion_matrix_dir, exist_ok=True)
        print(f"路径 {confusion_matrix_dir} 创建成功！")
    else:
        print(f"路径 {confusion_matrix_dir} 已存在，跳过创建操作。")

    # 3. 日志配置
    logging.basicConfig(filename=os.path.join(save_dir, 'training.log'), level=logging.INFO)
    print("日志初始化完成")

    # 创建 TensorBoard 日志对象
    writer = SummaryWriter(log_dir=save_dir)
    print("TensorBoard 日志创建成功！")

    # # 初始化日志记录
    # logging.basicConfig(filename=os.path.join(args.save_dir, 'training.log'), level=logging.INFO,
    #                         format='%(asctime)s - %(levelname)s - %(message)s')
    # 读取数据划分信息
    data_splits = read_data_split(args.data_split_file)
    logging.info(f"选择模型：PD_DiaCrossNet")
    logging.info(f"选择模态：Seg+QSM+T1")

    # 循环遍历每一折进行交叉验证
    for fold_idx in range(args.num_folds):
        print(f"Training fold {fold_idx }/{args.num_folds-1}")
        logging.info(f"Training fold {fold_idx}/{args.num_folds-1}")

        # 获取当前折的训练和验证文件列表
        train_files, val_files = data_splits[fold_idx]

        # 初始化双模态数据集
        # train_dataset = Dual_Dataset(files_path=train_files,mode1='T1', mode2='Seg',transform=transform)
        # val_dataset= Dual_Dataset(files_path=val_files,mode1='T1', mode2='Seg',transform=None)
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        #初始化单模态数据集
        #train_dataset = Single_Dataset3D(files_path=train_files,  mode='QSM', transform=transform)
        #val_dataset = Single_Dataset3D(files_path=val_files, mode='QSM', transform=None)

        #concat双模台初始化
        # train_dataset = DualDataset2(
        #     files_path=train_files,
        #     mode=('QSM', 'Seg'),  # 指定双模态
        #     transform=transform  # 训练集启用数据增强
        # )
        #
        # val_dataset = DualDataset2(
        #     files_path=val_files,
        #     mode=('QSM', 'Seg'),  # 同上
        #     transform=None  # 验证集不增强
        # )
        train_dataset = Dual_Dataset(
                    files_path=train_files,
                    mode1='Seg',
                    mode2='QSM',
                    fusion_method='none',  # 返回两个独立的模态
                    transform = transform
                )
        val_dataset = Dual_Dataset(
            files_path=val_files,
            mode1='Seg',
            mode2='QSM',
            fusion_method='none',  # 关键：使用concat方法
            transform=None
        )

        #初始化三模态数据集
        # 初始化三模态数据集
        # train_dataset = Triple_Dataset(
        #     files_path=train_files,
        #     mode1='Seg',
        #     mode2='QSM',
        #     mode3='T1',
        #     fusion_method='none',  # 关键：使用concat方法
        #     transform = transform
        # )
        # val_dataset = Triple_Dataset(
        #     files_path=val_files,
        #     mode1='Seg',
        #     mode2='QSM',
        #     mode3='T1',
        #     fusion_method='none',  # 关键：使用concat方法
        #     transform=None
        # )

        # train_dataset = Triple_Dataset(files_path=train_files,mode1='Seg', mode2='QSM',mode3='T1', transform=transform)
        # val_dataset = Triple_Dataset(files_path=val_files,mode1='Seg',mode2='QSM',mode3='T1',transform=None)


        # train_dataset = Patch_Dataset3D(
        #     files_list=train_files,
        #     target_size=(224, 224, 224),
        #     patch_size=(56, 56, 56),
        #     stride=(28, 28, 28)  # 50%重叠
        # )
        # val_dataset = Patch_Dataset3D(
        #     files_list=val_files,
        #     target_size=(224, 224, 224),
        #     patch_size=(56, 56, 56),
        #     stride=(28, 28, 28)  # 50%重叠
        # )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,pin_memory=True)

        # 初始化模型、损失函数、优化器
        # model=resnext50(num_classes=1, input_channels=3).to(device)
        model = mymodel().to(device)
        # model = vgg(in_channels=3)
        # model = densenet121_3d(in_channels=2, num_classes=1).to(device)#, growth_rate=16
        # model=seresnext50().to(device)
        # model = resnet(in_channels=2, num_classes=1).to(device)
        # model= AG_SE_ResNeXt50_model().to(device)
        # model = AG_SE_ResNeXt50_model_3().to(device)
        # print(model)
        # print(model.device)
        # criterion = nn.CrossEntropyLoss().to(device)
        # 记录模型信息和模态数量
        print('选择模型：Resnext50')
        print('选择模态：QSM+Seg+T1')


        criterion = FocalLoss().to(device)
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=1e-4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=5e-4)
        # 学习率调度器
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 每2个epoch调整一次学习率
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=25, eta_min=1e-7
        )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='max', factor=0.5, patience=3,
        #     verbose=True, min_lr=1e-6
        # )
        # scheduler = cosine_annealing_with_warmup(optimizer,
        #                                          warmup_epochs=5,  # 预热5个epoch
        #                                          total_epochs=args.num_epochs)  # 总epochs数

        # 存储训练和验证结果
        train_losses, val_losses = [], []
        train_aucs, val_aucs = [], []
        train_accs,val_accs=[],[]

        best_res = -float('inf')
        best_cm = None

        # 初始化早停类
        # early_stopping = EarlyStopping(patience=15, min_delta=0.01)

        # 训练循环
        for epoch in range(1,args.num_epochs+1):
            print(f"Epoch {epoch}/{args.num_epochs}")
            # 训练阶段
            train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device,epoch,writer= writer,modes=args.modes_num)
            train_losses.append(train_loss)
            train_aucs.append(train_metrics['auc_roc'])
            train_accs.append(train_metrics['accuracy'])
            print(f"Train Loss: {train_loss:.4f}, Train Metrics: {train_metrics}")
            logging.info(f"Epoch:{epoch},Train Loss: {train_loss:.4f}, Train Metrics: {train_metrics}")

            # 验证阶段
            val_loss, val_metrics, cm = val_one_epoch(model, val_loader, criterion, device,epoch,writer=writer, save_Path = save_dir,modes=args.modes_num)
            val_losses.append(val_loss)
            val_aucs.append(val_metrics['auc_roc'])
            val_accs.append(val_metrics['accuracy'])
            print(f"Val Loss: {val_loss:.4f}, Val Metrics: {val_metrics}")
            logging.info(f"Epoch:{epoch},Val Loss: {val_loss:.4f}, Val Metrics: {val_metrics},CM:{cm}")


            # 实时绘制曲线
            plot_curve('Loss', train_losses, val_losses, fold_idx, save_dir, x_values=range(1, epoch + 1))
            plot_curve('AUC_roc', train_aucs, val_aucs, fold_idx, save_dir, x_values=range(1, epoch + 1))
            plot_curve('Accuracy', train_accs, val_accs, fold_idx, save_dir, x_values=range(1, epoch + 1))

            # TensorBoard记录
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Train/AUC', train_metrics['auc_roc'], epoch)
            writer.add_scalar('Val/AUC', val_metrics['auc_roc'], epoch)
            writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)

            # 学习率调度
            scheduler.step()

            # 保存最佳模型
            # if val_metrics['auc'] > best_res:
            #     best_res = val_metrics['auc']
            #     best_cm = cm  # 保存最佳混淆矩阵
            #     torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_fold_{fold_idx}.pth"))
            #
            #     print(f"Saved best model for fold {fold_idx}")
            if val_metrics['auc_roc']+val_metrics['f1_score']> best_res:
                best_res = val_metrics['auc_roc']+val_metrics['f1_score']
                best_weights = model.state_dict()
                # 保存模型
                torch.save(best_weights, os.path.join(save_dir, f"best_model_fold_{fold_idx}.pth"))
                print(f"Saved best model for fold {fold_idx }")
            #     logging.info(f"Saved best model for fold {fold_idx} at epoch {epoch}")
                # 早停检查
                # 早停检查（基于验证集损失）
            # early_stopping(val_loss)  # 传入当前验证损失
            # if early_stopping.early_stop:
            #     print(f"Early stopping triggered at epoch {epoch} for fold {fold_idx}.")
            #     logging.info(f"Early stopping triggered at epoch {epoch} for fold {fold_idx}.")
            #     break  # 提前停止训练


        # 在所有 epoch 结束后绘制曲线
        plot_curve('Loss', train_losses, val_losses, fold_idx, save_dir)
        plot_curve('AUC', train_aucs, val_aucs, fold_idx, save_dir)
        plot_curve('Accuracy', train_accs, val_accs, fold_idx, save_dir)

    # 关闭 TensorBoard writer
    writer.close()


        # # 训练结束后，加载最佳模型并进行测试
        # print(f"Loading best model for fold {fold_idx + 1}")
        # model.load_state_dict(torch.load(os.path.join(save_dir, f"best_model_fold_{fold_idx + 1}.pth")))
        #
        # test_avg_loss, test_acc, test_precision,test_recall, test_f1, test_specificity, auc = evaluate(model, test_loader, criterion, device)
        # print(f"Test Loss: {test_avg_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test Specificity: {test_specificity:.4f},  AUC: {auc:.4f}")


if __name__ == "__main__":
    main()