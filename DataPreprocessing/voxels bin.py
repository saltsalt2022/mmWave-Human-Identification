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




# 引入所需的库
import glob  # 用于文件路径匹配
import os  # 用于操作文件和目录
import numpy as np  # 用于处理数组和数值计算
import csv  # 用于处理CSV文件
import time  # 用于计时
from bin_to_pointcloud import PointCloudProcessCFG, RawDataReader, frame2pointcloud ,bin2np_frame,reg_data,FrameConfig
parent_dir = r'Data_bin/Train/NO2/'
absolute_path = os.path.abspath(parent_dir)  # 转换为绝对路径
print(absolute_path)
sub_dirs=['hdt','lxb','lhc']
extract_path = r'deal_Data_bin/Train/'
# 体素化函数：将点云数据转化为体素网格表示
def voxalize(x_points, y_points, z_points, x, y, z, velocity):
    # 获取点云数据的最小值和最大值
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    z_max = np.max(z)
    z_min = np.min(z)

    # 根据分辨率计算每个体素的尺寸
    z_res = (z_max - z_min) / z_points
    y_res = (y_max - y_min) / y_points
    x_res = (x_max - x_min) / x_points

    # 初始化体素网格为全零数组
    pixel = np.zeros([x_points, y_points, z_points])

    # 初始化当前坐标
    x_current = x_min
    y_current = y_min
    z_current = z_min

    # 上一个坐标值
    x_prev = x_min
    y_prev = y_min
    z_prev = z_min

    # 初始化计数器
    x_count = 0
    y_count = 0
    z_count = 0

    # 遍历每个点，计算该点属于哪个体素网格
    for i in range(y.shape[0]):
        x_current = x_min
        x_prev = x_min
        x_count = 0
        done = False

        while x_current <= x_max and x_count < x_points and not done:
            y_prev = y_min
            y_current = y_min
            y_count = 0
            while y_current <= y_max and y_count < y_points and not done:
                z_prev = z_min
                z_current = z_min
                z_count = 0
                while z_current <= z_max and z_count < z_points and not done:
                    # 如果该点在当前体素范围内，则增加该体素的计数
                    if x[i] < x_current and y[i] < y_current and z[i] < z_current and \
                        x[i] >= x_prev and y[i] >= y_prev and z[i] >= z_prev:
                        pixel[x_count, y_count, z_count] += 1
                        done = True
                    # 更新z坐标
                    z_prev = z_current
                    z_current += z_res
                    z_count += 1
                # 更新y坐标
                y_prev = y_current
                y_current += y_res
                y_count += 1
            # 更新x坐标
            x_prev = x_current
            x_current += x_res
            x_count += 1

    return pixel
def get_data_from_points(point_clouds):

    """
    从点云数据中提取体素化表示并生成训练数据
    :param point_clouds: 点云数据，形状为 (帧数, 点数, 属性数)(350,128,6)
    :return: 训练数据，形状为 (样本数, 60, 10, 32, 32)
    """
    pixels = []

    # 遍历每一帧点云数据
    for frame in point_clouds:
        # 提取点云的 x, y, z 坐标和速度
        x_c = frame[:, 0]
        y_c = frame[:, 1]
        z_c = frame[:, 2]
        vel_c = frame[:, 3]

        # 体素化当前帧点云数据
        pix = voxalize(10, 32, 32, x_c, y_c, z_c, vel_c)
        pixels.append(pix)

    # 将所有帧的体素化数据转换为 NumPy 数组
    pixels = np.array(pixels)

    # 定义时间窗口大小和滑动步长
    frames_together = 60  # 每个样本包含的帧数
    sliding = 10          # 滑动窗口的步长

    train_data = []

    # 使用滑动窗口生成训练数据
    i = 0
    while i + frames_together <= pixels.shape[0]:
        local_data = []
        for j in range(frames_together):
            local_data.append(pixels[i + j])
        train_data.append(local_data)
        i = i + sliding

    # 转换为 NumPy 数组
    train_data = np.array(train_data)

    # 清理内存
    del pixels

    return train_data



# 从bin文件中获取点云数据并进行处理
def get_data_from_bin(file_path, total_frames, pointCloudProcessCFG):
    bin_reader = RawDataReader(file_path)
    point_clouds = []
    collected_frames = 0
    for frame_no in range(total_frames):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
        np_frame = bin2np_frame(bin_frame)
        pointCloud = frame2pointcloud(np_frame, pointCloudProcessCFG)

        if pointCloud.shape[0] == 0 or pointCloud.shape[1] == 0:
            print(f"Frame {frame_no}: Empty point cloud")
            point_clouds.append(pointCloud)
            collected_frames +=1
            continue  # 跳过空帧
                # 对点云数据进行处理
        pointCloud = np.transpose(pointCloud, (1, 0))  # 转置点云数据
        pointCloud = reg_data(pointCloud, 128)  # 对点云数据进行正则化处理

        print(f"Frame {frame_no}: PointCloud shape: {pointCloud.shape}")
        point_clouds.append(pointCloud)

    bin_reader.close()
    return np.array(point_clouds)

# 解析雷达数据文件并提取特征和标签
def parse_RF_files(parent_dir, sub_dirs, file_ext='*.bin'):
    print(sub_dirs)

    # 初始化特征数组和标签列表
    features = np.empty((0, 60, 10, 32, 32))  # 用于存储特征数据
    labels = []

    pointCloudProcessCFG = PointCloudProcessCFG()  # 点云处理配置
    total_frames = 350 # 假设每个bin文件包含100帧数据

    # 遍历每个子目录，提取数据
    for sub_dir in sub_dirs:
        files = sorted(glob.glob(os.path.join(parent_dir, sub_dir, file_ext)))  # 获取子目录中的所有bin文件
        print(f"Searching in directory: {os.path.join(parent_dir, sub_dir)}")
        print(f"Found files: {files}")
        for fn in files:
            print(fn)
            print(sub_dir)

            # 获取点云数据
            point_clouds = get_data_from_bin(fn, total_frames, pointCloudProcessCFG)
            # 保存点云数据

            # 将点云数据体素化
            train_data = get_data_from_points(point_clouds)
            features=np.vstack([features,train_data])

            for i in range(train_data.shape[0]):
                labels.append(sub_dir)
            print(features.shape,len(labels))

            del train_data

    return features, labels

# 遍历所有子目录，提取数据并保存
for sub_dir in sub_dirs:
    features, labels = parse_RF_files(parent_dir, [sub_dir])

    # 构造保存路径
    Data_path = extract_path + sub_dir
    # 检查目录是否存在，如果不存在则创建
    os.makedirs(os.path.dirname(Data_path), exist_ok=True)
    # 保存数据为npz文件
    np.savez(Data_path, features, labels)

    # 清理数据
    del features, labels
