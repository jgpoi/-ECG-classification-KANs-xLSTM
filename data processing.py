import numpy as np  # 用于数值计算
import wfdb  # 处理 MIT-BIH 心电信号数据
from scipy.signal import butter, filtfilt  # 信号处理相关工具
import pywt  # 小波变换库
import csv  # 处理 CSV 文件

# AAMI 心电信号分类映射，将 MIT-BIH 标签映射为 AAMI 分类标签
AAMI_MIT = {
    'N': 'N',  # 正常心拍
    'L': 'N',
    'R': 'N',
    'e': 'N',
    'j': 'N',
    'A': 'S',  # 房性异常心拍
    'J': 'S',
    'S': 'S',
    'V': 'V',  # 室性异常心拍
    'E': 'V',
    'F': 'F',  # 房室传导阻滞心拍
    '/': 'Q',  # 未分类的心拍
    'f': 'Q',
    'Q': 'Q'
}

# 数据文件名列表，对应 MIT-BIH 数据集中的记录文件
file_name = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# 归一化函数，将信号归一化到 [-1, 1] 范围
def normalize(signal):
    return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1

# 小波去噪函数，利用离散小波变换去噪
def wavelet_denoise(signal, wavelet='db4', level=5, threshold=0.2):
    coeffs = pywt.wavedec(signal, wavelet, level=level)  # 分解信号
    # 假设 threshold 是你已经计算好的阈值（比如 Donoho's universal threshold）
    coeffs_thresholded = [coeffs[0]]  # 保留 A5

    # 对 D5 ~ D1 进行软阈值处理
    for c in coeffs[1:]:
       c_thresh = pywt.threshold(c, threshold, mode='soft')
       coeffs_thresholded.append(c_thresh)

    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)  # 重构信号
    return denoised_signal[:len(signal)]  # 返回与原信号等长的结果

# 输出文件路径
output_file = "ECG_data.csv"

# 初始化 CSV 文件（写入表头）
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Signal", "Label"])  # 写入列名

# 逐文件处理并写入
for F_name in file_name:
    # 读取记录和注释
    annotation = wfdb.rdann(f'./MIT-BIH/{F_name}', "atr")  # 加载注释
    record = wfdb.rdrecord(f'./MIT-BIH/{F_name}', physical=True, channels=[0])  # 加载信号
    fs = record.fs  # 获取采样频率

    # 筛选有效标签并映射
    index = np.isin(annotation.symbol, list(AAMI_MIT.keys()))  # 筛选有效标签索引
    labels = np.array(annotation.symbol)[index]  # 获取有效标签
    samples = annotation.sample[index]  # 获取标签对应的采样点
    mapped_labels = [AAMI_MIT[label] for label in labels]  # 映射标签为 AAMI 类别

    # 处理数据并写入
    batch_data = []  # 保存当前文件的所有数据
    for sample, label in zip(samples, mapped_labels):
        # 检查截取范围有效性
        if 200 < sample < len(record.p_signal) - 400:
            signal_segment = record.p_signal[sample - 100:sample + 200, 0]  # 截取信号段
            denoised_signal = wavelet_denoise(signal_segment)  # 小波去噪
            normalized_signal = np.round(normalize(denoised_signal), decimals=6)  # 归一化
            batch_data.append([list(normalized_signal), label])  # 保存结果

    # 写入文件
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(batch_data)  # 写入批量数据

    print(f"File: {F_name}, Processing Completed.")  # 打印完成信息

print("数据处理完成，保存至", output_file)  # 处理结束
