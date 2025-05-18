import sys
import os
import struct
import time
import numpy as np
import array as arr
import configuration as cfg
from scipy.ndimage import convolve1d
import queue  # 导入队列模块，用于存储点云数据

# 定义一个函数用于从字节数据读取8个字节并解析为整数
def read8byte(x):
    return struct.unpack('<hhhh', x)

# 帧配置类，用于存储点云数据处理中的各种配置
class FrameConfig:
    def __init__(self):
        # 通过配置文件获取配置参数
        self.numTxAntennas = cfg.NUM_TX  # 发送天线数量
        self.numRxAntennas = cfg.NUM_RX  # 接收天线数量
        self.numLoopsPerFrame = cfg.LOOPS_PER_FRAME  # 每帧循环次数
        self.numADCSamples = cfg.ADC_SAMPLES  # ADC采样数
        self.numAngleBins = cfg.NUM_ANGLE_BINS  # 角度bin数量

        # 计算每帧中 chirp 的数量
        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = self.numADCSamples  # 范围bin数量
        self.numDopplerBins = self.numLoopsPerFrame  # 多普勒bin数量

        # 计算每个chirp的大小
        self.chirpSize = self.numRxAntennas * self.numADCSamples
        # 计算每个chirp循环的大小（TDM模式下每个loop包含多个chirp）
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        # 计算每帧的大小
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame

# 点云处理配置类，存储点云处理的各种配置选项
class PointCloudProcessCFG:
    def __init__(self):
        self.frameConfig = FrameConfig()  # 初始化帧配置
        self.enableStaticClutterRemoval = True  # 启用静态杂波移除
        self.EnergyTop128 = True  # 启用能量前128个点筛选
        self.RangeCut = True  # 启用范围裁剪
        self.outputVelocity = True  # 输出速度
        self.outputSNR = True  # 输出信噪比
        self.outputRange = True  # 输出范围
        self.outputInMeter = True  # 输出的单位为米

        dim = 3  # x, y, z 三个维度
        if self.outputVelocity:
            self.velocityDim = dim
            dim += 1
        if self.outputSNR:
            self.SNRDim = dim
            dim += 1
        if self.outputRange:
            self.rangeDim = dim
            dim += 1
        self.couplingSignatureBinFrontIdx = 5  # 前置耦合信号的索引
        self.couplingSignatureBinRearIdx = 4  # 后置耦合信号的索引
        self.sumCouplingSignatureArray = np.zeros(
            (self.frameConfig.numTxAntennas, self.frameConfig.numRxAntennas, self.couplingSignatureBinFrontIdx + self.couplingSignatureBinRearIdx),
            dtype=complex
        )  # 初始化耦合信号数组

# 用于读取原始数据（ADC二进制数据）
class RawDataReader:
    def __init__(self, path):
        self.path = path  # 设置文件路径
        self.ADCBinFile = open(path, 'rb')  # 以二进制模式打开文件

    # 获取下一帧数据
    def getNextFrame(self, frameconfig):
        frame = np.frombuffer(self.ADCBinFile.read(frameconfig.frameSize * 4), dtype=np.int16)  # 读取指定大小的帧数据
        return frame 

    def close(self):
        self.ADCBinFile.close()  # 关闭文件

# 将二进制数据转换为复数格式的numpy数组
def bin2np_frame(bin_frame):
    np_frame = np.zeros(shape=(len(bin_frame) // 2), dtype=complex)
    np_frame[0::2] = bin_frame[0::4] + 1j * bin_frame[2::4]  # 奇数位置数据
    np_frame[1::2] = bin_frame[1::4] + 1j * bin_frame[3::4]  # 偶数位置数据
    return np_frame

# 重新整理帧数据，重新排列为适合处理的形式
def frameReshape(frame, frameConfig):
    frameWithChirp = np.reshape(frame, (frameConfig.numLoopsPerFrame, frameConfig.numTxAntennas, frameConfig.numRxAntennas, -1))
    return frameWithChirp.transpose(1, 2, 0, 3)  # 转置以适应后续的FFT处理[Tx, Rx, Loops, ADC]

# 进行范围FFT，计算每个距离bin的FFT结果
def rangeFFT(reshapedFrame, frameConfig):
    windowedBins1D = reshapedFrame * np.hamming(frameConfig.numADCSamples)  # 应用汉明窗
    rangeFFTResult = np.fft.fft(windowedBins1D)  # 计算FFT
    return rangeFFTResult

# 静态杂波移除处理
def clutter_removal(input_val, axis=0):
    reordering = np.arange(len(input_val.shape))  # 获取输入数据的维度顺序
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)  # 重新排列维度
    mean = input_val.mean(0)  # 计算均值
    output_val = input_val - mean  # 从数据中减去均值（静态杂波移除）
    return output_val.transpose(reordering)  # 恢复原维度顺序

# 进行多普勒FFT处理
def dopplerFFT(rangeResult, frameConfig):
    windowedBins2D = rangeResult * np.reshape(np.hamming(frameConfig.numLoopsPerFrame), (1, 1, -1, 1))  # 应用汉明窗
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)  # 沿着多普勒维度计算FFT
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)  # 将FFT结果移到中心
    return dopplerFFTResult

# 计算点云的x, y, z坐标
def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64):
    assert num_tx > 2, "need a config for more than 2 TXs"
    num_detected_obj = virtual_ant.shape[1]  # 检测到的物体数量
    azimuth_ant = virtual_ant[:2 * num_rx, :]  # 获取方位角天线数据
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=complex)
    azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant  # 填充到合适的大小

    # 处理方位角信息
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.abs(azimuth_fft), axis=0)  # 获取最大值索引
    peak_1 = np.zeros_like(k_max, dtype=complex)
    for i in range(len(k_max)):
        peak_1[i] = azimuth_fft[k_max[i], i]
    k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
    wx = 2 * np.pi / fft_size * k_max
    x_vector = wx / np.pi
    # 处理仰角信息
    elevation_ant = virtual_ant[2 * num_rx:, :]
    elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=complex)
    elevation_ant_padded[:num_rx, :] = elevation_ant

    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)
    peak_2 = np.zeros_like(elevation_max, dtype=complex)
    for i in range(len(elevation_max)):
        peak_2[i] = elevation_fft[elevation_max[i], i]

    # 计算仰角相位
    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    ypossible = 1 - x_vector ** 2 - z_vector ** 2
    y_vector = ypossible
    x_vector[ypossible < 0] = 0
    z_vector[ypossible < 0] = 0
    y_vector[ypossible < 0] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector

# 将帧数据转换为点云
def frame2pointcloud(frame, pointCloudProcessCFG):
    frameConfig = pointCloudProcessCFG.frameConfig
    reshapedFrame = frameReshape(frame, frameConfig)  # 重新整理帧数据
    rangeResult = rangeFFT(reshapedFrame, frameConfig)  # 进行范围FFT

    if pointCloudProcessCFG.enableStaticClutterRemoval:
        rangeResult = clutter_removal(rangeResult, axis=2)  # 移除静态杂波

    dopplerResult = dopplerFFT(rangeResult, frameConfig)  # 进行多普勒FFT

    # 进行CFAR处理，识别出能量最大的目标
    dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0, 1))
    dopplerResultInDB = np.log10(np.absolute(dopplerResultSumAllAntenna))

    if pointCloudProcessCFG.RangeCut:  # 过滤掉范围过近或过远的点
        dopplerResultInDB[:, :25] = -100
        dopplerResultInDB[:, 125:] = -100

    # 使用能量阈值进行CFAR处理
    cfarResult = np.zeros(dopplerResultInDB.shape, bool)
    if pointCloudProcessCFG.EnergyTop128:
        top_size = 128
        energyThre128 = np.partition(dopplerResultInDB.ravel(), 128 * 128 - top_size - 1)[128 * 128 - top_size - 1]
        cfarResult[dopplerResultInDB > energyThre128] = True

    det_peaks_indices = np.argwhere(cfarResult == True)  # 获取检测到的目标的索引
    R = det_peaks_indices[:, 1].astype(np.float64)
    V = (det_peaks_indices[:, 0] - frameConfig.numDopplerBins // 2).astype(np.float64)

    if pointCloudProcessCFG.outputInMeter:
        R *= cfg.RANGE_RESOLUTION
        V *= cfg.DOPPLER_RESOLUTION

    energy = dopplerResultInDB[cfarResult == True]

    AOAInput = dopplerResult[:, :, cfarResult == True]
    AOAInput = AOAInput.reshape(12, -1)

    if AOAInput.shape[1] == 0:
        return np.array([]).reshape(6, 0)

    # 计算目标的三维位置
    x_vec, y_vec, z_vec = naive_xyz(AOAInput)
    
    x, y, z = x_vec * R, y_vec * R, z_vec * R
    pointCloud = np.concatenate((x, y, z, V, energy, R))
    pointCloud = np.reshape(pointCloud, (6, -1))  # 转换为标准格式
    pointCloud = pointCloud[:, y_vec != 0]  # 过滤掉无效点
    return pointCloud

# 数据正则化，将点云数据调整为指定大小
def reg_data(data, pc_size):
    pc_tmp = np.zeros((pc_size, 6), dtype=np.float32)
    pc_no = data.shape[0]
    if pc_no < pc_size:
        fill_list = np.random.choice(pc_size, size=pc_no, replace=False)
        fill_set = set(fill_list)
        pc_tmp[fill_list] = data
        dupl_list = [x for x in range(pc_size) if x not in fill_set]
        dupl_pc = np.random.choice(pc_no, size=len(dupl_list), replace=True)
        pc_tmp[dupl_list] = data[dupl_pc]
    else:
        pc_list = np.random.choice(pc_no, size=pc_size, replace=False)
        pc_tmp = data[pc_list]
    return pc_tmp

# 主程序入口
if __name__ == '__main__':
    bin_filename = sys.argv[1]  # 获取命令行传入的bin文件路径
    total_frame_number = int(sys.argv[2])  # 获取总帧数

    pointCloudProcessCFG = PointCloudProcessCFG()  # 初始化点云处理配置
    shift_arr = cfg.MMWAVE_RADAR_LOC  # 雷达位置偏移配置
    bin_reader = RawDataReader(bin_filename)  # 初始化读取器，打开bin文件

    q_pointcloud = queue.Queue()  # 创建队列用于存储点云数据
    collected_frames = 0  # 已处理的帧数

    # 循环处理每一帧数据
    for frame_no in range(total_frame_number):
        bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)  # 获取下一帧数据
        np_frame = bin2np_frame(bin_frame)  # 转换为复数形式的numpy数组
        pointCloud = frame2pointcloud(np_frame, pointCloudProcessCFG)  # 转换为点云数据

        if pointCloud.shape[0] == 0 or pointCloud.shape[1] == 0:  # 如果没有检测到目标，跳过当前帧
            q_pointcloud.put(raw_points)  # 将空点云数据放入队列
            collected_frames += 1
            continue  # 跳过当前帧，继续处理下一帧

        # 对点云数据进行处理
        raw_points = np.transpose(pointCloud, (1, 0))  # 转置点云数据
        raw_points[:, :3] = raw_points[:, :3] + shift_arr  # 添加雷达位置偏移
        raw_points = reg_data(raw_points, 128)  # 对点云数据进行正则化处理

        # 确保输出目录存在
        output_dir = 'bin_to_txt/NO1/'
        os.makedirs(output_dir, exist_ok=True)

        # 创建文件路径并保存点云数据
        output_file = f'{output_dir}pointcloud_frame_{frame_no}.txt'
        np.savetxt(output_file, raw_points, fmt='%.6f', header='x y z velocity energy range', comments='')

        # 打印当前帧的点云形状
        print(f'Frame {frame_no}: PointCloud shape: {raw_points.shape}')

        # 将处理后的点云数据放入队列
        q_pointcloud.put(raw_points)

    # 完成所有帧的处理后，关闭文件读取器
    bin_reader.close()
