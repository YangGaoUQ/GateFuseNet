import nibabel as nib
import numpy as np
import torch
from nibabel.imageglobals import LoggingOutputSuppressor
from torch.utils.data import Dataset
from scipy import ndimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Triple_Dataset(Dataset):
    def __init__(self, files_path, mode1='QSM', mode2='T1', mode3='Seg', transform=None):
        """
        三模态3D医学图像数据集类
        参数：
            files_path: 文件路径列表
            mode1: 第一种模态 ('QSM', 'T1', 'Seg')
            mode2: 第二种模态 ('QSM', 'T1', 'Seg')
            mode3: 第三种模态 ('QSM', 'T1', 'Seg')
            transform: 数据增强变换
        """
        self.files_path = files_path
        self.mode1 = mode1
        self.mode2 = mode2
        self.mode3 = mode3
        self.transform = transform
        self.target_size = (128, 128, 128)  # (D, H, W)
        self.target_spacing = (1.0, 1.0, 1.0)  # 目标分辨率 1×1×1mm



    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        file_line = self.files_path[idx]
        files = file_line.split()

        # 获取三种模态的文件路径
        file_path1 = self._get_file_path(files, self.mode1)
        file_path2 = self._get_file_path(files, self.mode2)
        file_path3 = self._get_file_path(files, self.mode3)

        # 加载并处理三种模态
        img1 = self._load_and_process_modality(file_path1, self.mode1)
        img2 = self._load_and_process_modality(file_path2, self.mode2)
        img3 = self._load_and_process_modality(file_path3, self.mode3)

        # 应用相同的数据增强（保持空间对齐）
        if self.transform:
            seed = torch.randint(0, 2 ** 32, (1,)).item()

            torch.manual_seed(seed)
            img1 = self.transform(img1)

            torch.manual_seed(seed)
            img2 = self.transform(img2)

            torch.manual_seed(seed)
            img3 = self.transform(img3)

        label = int(files[-1])
        return (img1, img2, img3), torch.tensor(label, dtype=torch.long)

    def _get_file_path(self, files, modality):
        """根据模态获取对应的文件路径"""
        if modality == 'QSM':
            return files[0]
        elif modality == 'T1':
            return files[1]
        elif modality == 'Seg':
            return files[2]
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def _get_voxel_spacing_from_header(self, nii_img):
        """从NIfTI文件的仿射变换矩阵中获取体素间距"""
        # 从仿射变换矩阵中提取体素尺寸
        affine = nii_img.affine
        voxel_size = np.abs(np.diag(affine)[:3])
        # 返回(z, y, x)顺序的体素间距，对应转置后的数据轴序
        return (float(voxel_size[2]), float(voxel_size[1]), float(voxel_size[0]))

    def _load_and_process_modality(self, file_path, modality):
        """加载并处理单个模态数据"""
        # 加载NIfTI文件
        nii_img = nib.load(file_path)
        img = nii_img.get_fdata()
        img = np.transpose(img, (2, 1, 0))  # 转为(z,y,x)

        # 从文件头获取真实的体素间距
        original_spacing = self._get_voxel_spacing_from_header(nii_img)

        # print(f"加载 {modality} 数据，原始尺寸: {img.shape}, 原始间距: {original_spacing}")

        # 检查是否需要重采样
        if not self._is_spacing_equal(original_spacing, self.target_spacing):
            img = self._resample_to_target_spacing(img, original_spacing)
            # print(f"{modality} 重采样后尺寸: {img.shape}")


        # # 各向异性重采样 (QSM/Seg)
        # if modality in ['QSM', 'Seg']:
        #     img = self.__resample_isotropic__(img, self.voxel_size[modality])

        # 中心裁剪
        img = self._center_crop_or_pad(img)

        # 模态特定归一化
        img = self.__modality_specific_normalization__(img, modality)

        return torch.from_numpy(img).float().unsqueeze(0)  # (1, D, H, W)

    def _is_spacing_equal(self, spacing1, spacing2, tolerance=1e-3):
        """检查两个间距是否相等（考虑浮点数误差）"""
        return all(abs(s1 - s2) < tolerance for s1, s2 in zip(spacing1, spacing2))

    def _resample_to_target_spacing(self, data, original_spacing):
        """将数据从原始间距重采样到目标间距"""
        # 计算重采样比例
        zoom_factors = [
            original_spacing[i] / self.target_spacing[i]
            for i in range(3)
        ]

        # print(f"重采样比例: {zoom_factors}")

        # order=3表示使用三次样条插值
        resampled_data = ndimage.zoom(data, zoom_factors, order=3)

        return resampled_data

    def _center_crop_or_pad(self, data):
        """中心裁剪或填充到目标尺寸"""
        target_d, target_h, target_w = self.target_size
        depth, height, width = data.shape

        # 首先处理填充
        if depth < target_d or height < target_h or width < target_w:
            pad_d = max(0, target_d - depth)
            pad_h = max(0, target_h - height)
            pad_w = max(0, target_w - width)

            data = np.pad(data, (
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2)
            ), mode='constant', constant_values=0)

            depth, height, width = data.shape
            # print(f"填充后尺寸: {data.shape}")

        # 然后处理裁剪
        if depth > target_d or height > target_h or width > target_w:
            # 计算裁剪起始位置
            start_d = max(0, (depth - target_d) // 2)
            start_h = max(0, (height - target_h) // 2)
            start_w = max(0, (width - target_w) // 2)

            # 确保不会越界
            end_d = min(depth, start_d + target_d)
            end_h = min(height, start_h + target_h)
            end_w = min(width, start_w + target_w)

            data = data[start_d:end_d, start_h:end_h, start_w:end_w]
            # print(f"裁剪后尺寸: {data.shape}")

        return data

    def __modality_specific_normalization__(self, data, modality):
        """模态特定的归一化"""
        if modality == 'QSM':
            # QSM数据使用99百分位数归一化
            non_zero_data = data[data != 0]
            if len(non_zero_data) > 0:
                abs_percentile = np.percentile(np.abs(non_zero_data), 99)
                if abs_percentile > 0:
                    return np.clip(data / (abs_percentile + 1e-6), -1, 1)
            return data

        elif modality == 'T1':
            # T1数据使用脑区域的均值和标准差归一化
            brain_mask = data > data.mean()
            if np.sum(brain_mask) > 0:
                brain_mean = data[brain_mask].mean()
                brain_std = data[brain_mask].std()
                if brain_std > 0:
                    return (data - brain_mean) / brain_std
            return data

        elif modality == 'Seg':
            # Seg数据使用与QSM相同的归一化方法
            non_zero_data = data[data != 0]
            if len(non_zero_data) > 0:
                abs_percentile = np.percentile(np.abs(non_zero_data), 99)
                if abs_percentile > 0:
                    return np.clip(data / (abs_percentile + 1e-6), -1, 1)
            return data

        else:
            raise ValueError(f"Unknown modality: {modality}")

    # def __resample_isotropic__(self, data, original_spacing):
    #     """将各向异性数据重采样为1mm各向同性"""
    #     new_size = [
    #         int(data.shape[0] * original_spacing[0]),
    #         int(data.shape[1] * original_spacing[1]),
    #         int(data.shape[2] * original_spacing[2])
    #     ]
    #     scale = [
    #         new_size[0] / data.shape[0],
    #         new_size[1] / data.shape[1],
    #         new_size[2] / data.shape[2]
    #     ]
    #     return ndimage.zoom(data, scale, order=3)

    # def __center_crop__(self, data):
    #     """中心裁剪到目标尺寸"""
    #     target_d, target_h, target_w = self.target_size
    #     depth, height, width = data.shape
    #
    #     # 如果图像尺寸小于目标尺寸，进行填充
    #     if depth < target_d or height < target_h or width < target_w:
    #         pad_d = max(0, target_d - depth)
    #         pad_h = max(0, target_h - height)
    #         pad_w = max(0, target_w - width)
    #
    #         data = np.pad(data, (
    #             (pad_d // 2, pad_d - pad_d // 2),
    #             (pad_h // 2, pad_h - pad_h // 2),
    #             (pad_w // 2, pad_w - pad_w // 2)
    #         ), mode='constant', constant_values=0)
    #
    #         depth, height, width = data.shape
    #         print(f"填充后尺寸: {data.shape}")
    #
    #     # 计算裁剪起始位置
    #     start_d = (depth - target_d) // 2
    #     start_h = (height - target_h) // 2
    #     start_w = (width - target_w) // 2
    #
    #     # 确保索引不为负
    #     start_d = max(0, start_d)
    #     start_h = max(0, start_h)
    #     start_w = max(0, start_w)
    #
    #     return data[
    #            start_d:start_d + target_d,
    #            start_h:start_h + target_h,
    #            start_w:start_w + target_w
    #            ]
        # start_d = (depth - target_d) // 2
        # start_h = (height - target_h) // 2
        # start_w = (width - target_w) // 2
        #
        # return data[
        #        start_d:start_d + target_d,
        #        start_h:start_h + target_h,
        #        start_w:start_w + target_w
        #        ]


    # def __modality_specific_normalization__(self, data, modality):
    #     """模态特定的归一化"""
    #     if modality == 'QSM':
    #         abs_percentile = np.percentile(np.abs(data[data > 0]), 99)
    #         return np.clip(data / (abs_percentile + 1e-6), -1, 1)
    #     elif modality == 'T1':
    #         brain_mask = data > data.mean()
    #         return (data - data[brain_mask].mean()) / data[brain_mask].std()
    #     elif modality == 'Seg':
    #         # 获取体积图像中所有非零像素
    #         # pixels = data[data > 0]
    #         # # 计算非零区域的均值和标准差
    #         # mean = pixels.mean()
    #         # std = pixels.std()
    #         # # 根据均值和标准差对体积图像进行标准化
    #         # out = (data - mean) / std
    #         # # 创建与输入体积相同形状的随机噪声，均值为0，标准差为1
    #         # out_random = np.random.normal(0, 1, size=data.shape)
    #         # # 对于原体积图像中值为零的位置，用随机噪声填充
    #         # out[data == 0] = out_random[data == 0]
    #         # 返回归一化后的体积图像
    #         abs_percentile = np.percentile(np.abs(data[data > 0]), 99)
    #         return np.clip(data / (abs_percentile + 1e-6), -1, 1)
    #         # return data原来只有这一个
    #     else:
    #         raise ValueError(f"Unknown modality: {modality}")

