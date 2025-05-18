"""
Extract the VOXEL representation from the point cloud dataset

USAGE: change the parent_dir and extract_path to the correct variables.

- parent_dir is raw Path_to_training_or_test_data.
- extract_path is the Path_to_put_extracted_data samples.

EXAMPLE: SPECIFICATION

parent_dir = '/Users/sandeep/Research/Ti-mmWave/data/Temp_Samples/Train/'
sub_dirs=['boxing','jack','jump','squats','walk']
extract_path = '/Users/sandeep/Research/Ti-mmWave/data/extract/Train_Data_voxels_'
"""

# 输入数据的上级目录
parent_dir = 'Data/Train/'
# 子目录，分别是各类动作
sub_dirs=['boxing','jack','jump','squats','walk']
# 接收处理后数据的路径
extract_path = 'deal_Data/Train/'

import glob
import os
import numpy as np
import csv
import time

# 将点云数据转换为 3D voxel grid
# x_points, y_points, z_points 是 voxel 维度的样本数，x/y/z 是空间坐标
# velocity 是速度值

def voxalize(x_points, y_points, z_points, x, y, z, velocity):
    # 计算 xyz 范围
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    z_max = np.max(z)
    z_min = np.min(z)

    # 计算维度粒度
    z_res = (z_max - z_min)/z_points
    y_res = (y_max - y_min)/y_points
    x_res = (x_max - x_min)/x_points

    # 初始化空间 voxel 网格
    pixel = np.zeros([x_points,y_points,z_points])

    # 对每个点扫描定位在哪个 voxel 中
    for i in range(y.shape[0]):
        x_current = x_min
        x_prev = x_min
        x_count = 0
        done=False

        while x_current <= x_max and x_count < x_points and not done:
            y_prev = y_min
            y_current = y_min
            y_count = 0
            while y_current <= y_max and y_count < y_points and not done:
                z_prev = z_min
                z_current = z_min
                z_count = 0
                while z_current <= z_max and z_count < z_points and not done:
                    if x[i] < x_current and y[i] < y_current and z[i] < z_current and x[i] >= x_prev and y[i] >= y_prev and z[i] >= z_prev:
                        # 统计该 voxel 内的点数
                        pixel[x_count,y_count,z_count] += 1
                        done = True
                    z_prev = z_current
                    z_current += z_res
                    z_count += 1
                y_prev = y_current
                y_current += y_res
                y_count += 1
            x_prev = x_current
            x_current += x_res
            x_count += 1
    return pixel

# 从 txt 文件中读取完整的点云数据

def get_data(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    # 初始化数据存储结构
    frame_num_count = -1
    frame_num = []
    x, y, z, velocity, intensity = [], [], [], [], []
    wordlist = []

    # 解析每行的文本内容
    for x1 in lines:
        for word in x1.split():
            wordlist.append(word)

    length1 = len(wordlist)

    for i in range(0,length1):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            frame_num_count += 1
        if wordlist[i] == "point_id:":
            frame_num.append(frame_num_count)
        if wordlist[i] == "x:":
            x.append(wordlist[i+1])
        if wordlist[i] == "y:":
            y.append(wordlist[i+1])
        if wordlist[i] == "z:":
            z.append(wordlist[i+1])
        if wordlist[i] == "velocity:":
            velocity.append(wordlist[i+1])
        if wordlist[i] == "intensity:":
            intensity.append(wordlist[i+1])

    # 转换成 numpy 数组并输入数据类型
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    frame_num = np.asarray(frame_num, dtype=int)

    # 根据帧编号分组
    data = dict()
    for i in range(len(frame_num)):
        if frame_num[i] in data:
            data[frame_num[i]].append([x[i], y[i], z[i], velocity[i], intensity[i]])
        else:
            data[frame_num[i]] = [[x[i], y[i], z[i], velocity[i], intensity[i]]]

    # 将多帧合并为一个时间窗口，并滑动接龙
    data_pro1 = dict()
    together_frames = 1
    sliding_frames = 1
    frames_number = sorted(data.keys())
    total_frames = max(frames_number)

    i = j = 0
    while together_frames + i < total_frames:
        curr_j_data = []
        for k in range(together_frames):
            curr_j_data += data[i + k]
        data_pro1[j] = curr_j_data
        j += 1
        i += sliding_frames

    pixels = []
    for i in data_pro1:
        f = np.array(data_pro1[i])
        x_c, y_c, z_c, vel_c = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
        pix = voxalize(10, 32, 32, x_c, y_c, z_c, vel_c)
        pixels.append(pix)

    pixels = np.array(pixels)

    # 按 60 帧一组，每 10 帧滑动,样本1: 帧  0 ~ 59,样本2: 帧 10 ~ 69,滑动10是为了丰富样本数量
    frames_together = 60
    sliding = 10
    train_data = []
    i = 0
    while i + frames_together <= pixels.shape[0]:
        local_data = []
        for j in range(frames_together):
            local_data.append(pixels[i + j])
        train_data.append(local_data)
        i += sliding

    return np.array(train_data)

# 解析不同动作目录下所有 txt 文件，输出 features + labels

def parse_RF_files(parent_dir, sub_dirs, file_ext='*.txt'):
    print(sub_dirs)
    features = np.empty((0, 60, 10, 32, 32))
    labels = []

    for sub_dir in sub_dirs:
        files = sorted(glob.glob(os.path.join(parent_dir, sub_dir, file_ext)))
        for fn in files:
            print(fn)
            print(sub_dir)
            train_data = get_data(fn)
            features = np.vstack([features, train_data])
            for i in range(train_data.shape[0]):
                labels.append(sub_dir)
            print(features.shape, len(labels))
            del train_data

    return features, labels

# 开始处理各个动作的文件

for sub_dir in sub_dirs:
    features, labels = parse_RF_files(parent_dir, [sub_dir])
    Data_path = extract_path + sub_dir
    os.makedirs(os.path.dirname(Data_path), exist_ok=True)
    np.savez(Data_path, features, labels)
    del features, labels
