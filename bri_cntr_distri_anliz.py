import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde
import numpy as np
import scipy.stats as stats


def get_brightness_list(folder):
    """
    遍历文件夹内所有图片，返回亮度（均值）列表
    支持 jpg/png/jpeg/bmp/tif
    """
    brightness = []
    contrast = []
    fixed_mean = 127.5
    for root, _, files in os.walk(folder):
        for f in tqdm(files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    brightness.append(float(img.mean()))
                    contrast.append(np.sqrt(np.mean((img - fixed_mean) ** 2)))
    return brightness, contrast

# -------------------------------------------------
# 2. 绘制直方图
# -------------------------------------------------
def hisplot(brightness_A, brightness_B):
    plt.figure(figsize=(8, 5))
    # 自动计算合理的 bin 数（可用 Sturges 规则或固定值）
    bins = 50
    plt.hist(brightness_A, bins=bins, alpha=0.6, label='Dataset A', color='steelblue', edgecolor='black')
    plt.hist(brightness_B, bins=bins, alpha=0.6, label='Dataset B', color='coral', edgecolor='black')
    plt.xlabel('Brightness (mean pixel value)')
    plt.ylabel('Number of Images')
    plt.title('Brightness Distribution Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ---------- 生成密度曲线 ----------
def PDFPlot(brightness_A, brightness_B):
    # 1. 计算 KDE
    kde_A = gaussian_kde(brightness_A)
    kde_B = gaussian_kde(brightness_B)

    # 2. 生成横坐标（亮度范围）
    x_min = min(brightness_A + brightness_B)
    x_max = max(brightness_A + brightness_B)
    x = np.linspace(x_min, x_max, 500)

    # 3. 计算密度
    y_A = kde_A(x)
    y_B = kde_B(x)

    # ---------- 绘制 ----------
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_A, label='Dataset A', color='steelblue', linewidth=2)
    plt.plot(x, y_B, label='Dataset B', color='coral',   linewidth=2)

    plt.xlabel('Brightness (mean pixel value)')
    plt.ylabel('Density')
    plt.title('Brightness Distribution Comparison (Line Plot)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# 方差齐性检验
def LevenesTest(brightness_A, brightness_B):
    # brightness_A 与 brightness_B 为已采集的列表或一维数组
    stat, p_value = stats.levene(brightness_A, brightness_B, center='median')

    print(f"Levene 统计量 = {stat:.4f}, p 值 = {p_value:.4g}")

    if p_value < 0.05:
        print("拒绝原假设：两组方差不齐")
    else:
        print("不拒绝原假设：两组方差齐")


if __name__ == '__main__':
    # -------------------------------------------------
    # 1. 数据集路径（请替换为您的真实文件夹）
    # -------------------------------------------------
    dataset_Nor_dir = "G:\\PaperCode\\zDataSets\\Pneumonia\\chest_xray\\NORMAL"
    dataset_Pne_dir = "G:\\PaperCode\\zDataSets\\Pneumonia\\chest_xray\\PNEUMONIA"
    save_path = "G:\\PaperCode\\zDataSets\\Pneumonia\\chest_xray\\brightness_comparison.png"
    brightness_Nor, contrast_nor = get_brightness_list(dataset_Nor_dir)
    brightness_Pne, contrast_Pne = get_brightness_list(dataset_Pne_dir)
    # 直方图
    # hisplot(brightness_Nor, brightness_Pne)
    # hisplot(contrast_nor, contrast_Pne)
    # 概率密度图
    # PDFPlot(brightness_Nor, brightness_Pne)
    PDFPlot(contrast_nor, contrast_Pne)
    LevenesTest(brightness_Nor, brightness_Pne)
    LevenesTest(contrast_nor, contrast_Pne)

