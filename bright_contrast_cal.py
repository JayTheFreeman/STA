import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd


def equal_freq_binning(data, n_bins=16):
    """
    对一维数据data进行等频分箱，返回n_bins个类别对应的标签（0到n_bins-1）和分箱边界。
    """
    data = np.array(data)
    # 计算分位数点
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(data, percentiles)

    # 分配标签
    labels = np.digitize(data, bin_edges[1:-1], right=False)
    # 确保没有标签超出范围（由于浮点精度，极端值可能出错）
    labels = np.clip(labels, 0, n_bins - 1)

    return labels, bin_edges

def bri_con_cal(file_list, image_dir):
    # 初始化一个列表来存储结果
    brightness_list = []
    contrast_list = []

    # 遍历所有图片
    for filename in tqdm(file_list):
        image_path = os.path.join(image_dir, filename)
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('L')  # 转换为灰度图
            img_array = np.array(img)

            # 计算亮度（平均像素值）
            brightness = np.mean(img_array)
            brightness_list.append(brightness)

            # 计算对比度（像素值的标准差）
            # contrast = np.std(img_array)
            fixed_mean = 127.5
            contrast = np.sqrt(np.mean((img_array - fixed_mean) ** 2))
            contrast_list.append(contrast)

        else:
            print(f"文件 {filename} 不存在，跳过。")

    # 对数据进行分箱
    brightness_labels, brightness_bin_edges = equal_freq_binning(brightness_list, n_bins=10)
    contrast_labels, contrast_bin_edges = equal_freq_binning(contrast_list, n_bins=10)
    results = [[a, b, c, d, e] for a, b, c, d, e in zip(file_list, brightness_list, contrast_list,
                                                        brightness_labels.tolist(),contrast_labels.tolist())]
    # print('bri edges {}'.format(brightness_bin_edges))
    # print('contrast edges {}'.format(contrast_bin_edges))
    return results


if __name__ == '__main__':
    # 定义图片文件夹路径和 CSV 文件路径
    image_dir = 'dataset'
    # 因为是基于data.csv文件逐个计算亮度、对比度的，所以最终得到的image_brightness_contrast.csv中各元素顺序和data.csv中一致。
    csv_file = 'dataset/data.csv'

    # 读取 CSV 文件中的图片名
    data = pd.read_csv(csv_file)
    file_list = data['Filename'].tolist()
    # 计算亮度和对比度
    results = bri_con_cal(file_list, image_dir)

    # 将结果保存到 CSV 文件中
    results_df = pd.DataFrame(results, columns=['Filename', 'Brightness', 'Contrast', 'Brightness_classes', 'Contrast_classes'])
    results_df.to_csv('dataset\\image_brightness_contrast.csv', index=False)

    print("亮度和对比度计算完成，结果已保存到 image_brightness_contrast.csv 文件中。")