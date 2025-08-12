import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

def generate_seg_data():
    # 读取excel文件中的数据
    file_path = 'AAL3_cluster2.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
    print(df.head)

    # 筛选第五列值为1的行
    filtered_df = df[df.iloc[:, 4] == 1]
    print(filtered_df)

    # 取需要分割的值，即第一列数字
    values = filtered_df.iloc[:, 0].values
    print(values)
    return values

#分割.nii图像
def seg_nii(nii_path,output_path,values):
    # 读取.nii图像
    nii_image=nib.load(nii_path)
    img_data=nii_image.get_fdata()

    #创建一个新的图像数组，初始值全为0（黑色）
    seg_data=np.zeros_like(img_data)

    # 将所有在values列表中的值对应的像素设为1
    seg_data[np.isin(img_data, values)] = 1
    # gray_value=250
    # for v in values:
    #     seg_data[img_data==v]=gray_value
    #     gray_value-=8


    # seg_data[np.isin(img_data,values)]=1

    #创建一个新的NIFTI图像
    new_img=nib.Nifti1Image(seg_data,nii_image.affine)

    #保存分割后的新图像
    nib.save(new_img,output_path)


def process_all_files(root_dir,values):
    "读取数据文件并处理图像"
    for folder_name in os.listdir(root_dir):
        folder_path=os.path.join(root_dir,folder_name)      #PD、HC
        if os.path.isdir(folder_path):
            print(folder_name)

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if not os.path.isdir(file_path):
                    continue

                atlas_file_name=f"Atlas_{file_name}.nii"
                atlas_path=os.path.join(file_path,atlas_file_name)
                print(atlas_path)

                if os.path.exists(atlas_path):
                    print(f'Processing{atlas_path}')
                    #分割图像并保存

                    output_filename=f"{file_name}_seg2.nii"
                    output_path=os.path.join(file_path,output_filename)
                    print(output_path)

                    seg_nii(atlas_path,output_path,values)
                else:
                    print(f"{atlas_path} does not exist.")
        else:
            print(f"{folder_path} is not a directory.")


if __name__=="__main__":
    seg_values=generate_seg_data()
    print(seg_values)
    root_dir='/home/sunwenwen/JR/PD_classfication/PD_Data'
    # root_dir='PD_Data'
    process_all_files(root_dir,seg_values)

    print("处理成功!")





    #

