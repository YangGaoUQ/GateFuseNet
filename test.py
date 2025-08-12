import logging
import os

import numpy as np
import torch

from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

from dataset.TripleDataset2 import Triple_Dataset

from models.AG_SE_ResNeXt50_3_modes import AG_SE_ResNeXt50_model_3
from models.DenseFormer import MultiStageDenseFormerTri
from models.DenseNet_3D import densenet121_3d
from train_val import parse_args
from utils.losses import FocalLoss
from utils.metrics import evaluate_metrics, plot_confusion_matrix, plot_roc_curve
from utils.my_utils import read_test_data, set_random_seed, load_best_model, save_results_to_excel
from models.ResNet50_3D_v1 import resnet50 as resnet
# from models.AG_SE_ResNeXt50 import AG_SE_ResNeXt50_model as mymodel
# os.chdir("/home/sunwenwen/JR/PD_classfication")
# print("CWD: ", os.getcwd())


from models.PD_AMF_c2 import create_pd_amfnet as mymodel
from models.ResNeXt_3D import resnext50_32x4d_3d as resnext50
from models.AG_SE_ResNeXt50_3 import AG_SE_ResNeXt50_model_3 as mymodel


def test(model, test_loader, device, writer=None, epoch=0, class_names=None, save_path=None, modes=1):
    if class_names is None:
        class_names = ['HC', 'PD']
    model.eval()  # 设置模型为评估模式

    all_labels = []
    all_preds = []
    all_probs = []  # 存储预测的概率值

    with torch.no_grad():
        for data in test_loader:
            if modes == 1:  # 单模态输入：(inputs, labels)
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1).float()
                outputs = model(inputs)

            elif modes == 2:  # 双模态输入：((img_data1, img_data2), labels)
                if isinstance(data, list) and len(data) == 2:
                    inputs, labels = data
                    if isinstance(inputs, list) and len(inputs) == 2:
                        img_data1, img_data2 = inputs
                        img_data1 = img_data1.to(device)
                        img_data2 = img_data2.to(device)
                        labels = labels.to(device)
                        labels = labels.unsqueeze(1).float()
                        outputs = model(img_data1, img_data2)
            elif modes == 3:  # 三模态输入：((img_data1, img_data2, img_data3), labels)
                (img_data1, img_data2, img_data3), labels = data
                img_data1 = img_data1.to(device)
                img_data2 = img_data2.to(device)
                img_data3 = img_data3.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1).float()
                outputs = model(img_data1, img_data2, img_data3)

            else:
                raise ValueError("数据格式不支持：必须是 (inputs, labels) 或 ((img_data1, img_data2), labels)")


            if outputs.shape[1] == 2:  # 二分类输出
                probs = torch.softmax(outputs, dim=1)[:, 1]  # 取正类（PD）的概率
            else:  
                probs = torch.sigmoid(outputs).squeeze(1)

            preds = (probs > 0.5).int()  # 生成硬标签

            # 🔥 关键修改：确保数据形状一致
            batch_labels = labels.squeeze().cpu().numpy()  # 去掉多余维度，变成1D
            batch_preds = preds.cpu().numpy()  
            batch_probs = probs.cpu().numpy() 

            # 保存当前批次的标签、预测和概率
            all_labels.extend(batch_labels)
            all_preds.extend(batch_preds)
            all_probs.extend(batch_probs)

    # 将 list 转换为 ndarray
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 添加调试信息
    print(f"Final shapes - Labels: {all_labels.shape}, Preds: {all_preds.shape}, Probs: {all_probs.shape}")

    # 计算评估指标
    metrics = evaluate_metrics(all_labels, all_preds, all_probs)
    plot_roc_curve(all_labels, all_probs)

    # 打印各项评估指标
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # 混淆矩阵绘制
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, writer, epoch, tag='Test_Confusion_Matrix', save_path=save_path)
    print(cm)
    return metrics, all_labels, all_probs



def main():
    set_random_seed(42)  # 设置固定的随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = parse_args()# 解析命令行参数

    test_file = 'test_files.txt'
    test_samples = read_test_data(test_file)

    # 加载最佳模型进行测试
    # save_dir = args.save_dir
    p1=os.path.join(os.getcwd(),'Output')
    save_dir=os.path.join(p1,'GateFuseNet')

    # 读取测试数据集
    # test_dataset = Single_Dataset3D(files_path=test_samples, mode='Seg', transform=None)

    #test_dataset=Dual_Dataset(files_path=test_samples, mode1='T1', mode2='Seg', transform=None)
    # test_dataset=DualDataset2(files_path=test_samples,mode=('QSM', 'Seg'), transform=None)
    test_dataset = Triple_Dataset(
        files_path=test_samples,
        mode1='Seg',
        mode2='QSM',
        mode3='T1',
        fusion_method='none', 
        transform=None
    )


    # test_dataset = Triple_Dataset(files_path=test_samples, mode1='Seg', mode2='QSM', mode3='T1', transform=None)
    #test_dataset=Triple_Dataset(files_path=test_samples, mode1='QSM',mode2='Seg',mode3='T1', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=2,pin_memory=True)

    # 初始化模型并加载最佳权重

    model = mymodel().to(device)

    best_model_paths = [os.path.join(save_dir, f"best_model_fold_{i}.pth") for i in range(5)]


    # 对每一个折的最优模型进行测试
    all_metrics = []
    all_labels = []
    all_probs = []
    results={"folds":{},"average":{}}

    fold_idx=0
    for model_path in best_model_paths:
        model = load_best_model(model, model_path)
        # criterion = FocalLoss().to(device)

        # 测试这个折的模型
        print(f"Testing with model from {model_path}")
        save_path = os.path.join(save_dir, f"Test/Confusion_Matrix_fold_{fold_idx}")
        metrics,labels,probs = test(model, test_loader, device,save_path=save_path,modes=args.modes_num)

        all_labels.extend(labels)
        all_probs.extend(probs)

        #保存每个折的结果
        fold_metrics=metrics.copy()
        all_metrics.append(fold_metrics)
        results["folds"][f"fold_{len(results['folds']) + 1}"] = fold_metrics

        fold_idx+=1

    # 计算五折的平均结果
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = np.mean([metrics[metric] for metrics in all_metrics])

    results["average"] = avg_metrics

    # 打印五折的平均结果
    for metric, value in avg_metrics.items():
        print(f"Average {metric.capitalize()} across 5 folds: {value:.4f}")

    # 输出并记录 all_labels 和 all_probs
    print("All Labels:", all_labels)
    print("All Probabilities:", all_probs)
    logging.info(f"All Labels: {all_labels}")
    logging.info(f"All Probabilities: {all_probs}")

    plot_roc_curve(all_labels, all_probs,save_path=os.path.join(save_dir, 'Test/ROC_Curves'))

    # 保存结果到Excel文件
    file_path=os.path.join(save_dir, "res.xlsx")
    save_results_to_excel(results, file_path)
    # save_results_to_excel(results, os.path.join(save_dir, f"Test/Confusion_Matrix_fold_{fold_idx}")"{save_dir}/model_results_PD.xlsx")


if __name__ == "__main__":
    main()
