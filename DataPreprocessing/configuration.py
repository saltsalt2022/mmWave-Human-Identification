import numpy as np

# 配置发射天线数量
NUM_TX = 3  # tx order tx0, tx2, tx1，面向板子（左，右，上）
NUM_RX = 4  # 接收天线数量

# 雷达配置的基本参数
START_FREQ = 60  # 起始频率（单位：GHz），即雷达的工作频率
ADC_START_TIME = 6  # ADC开始时间
FREQ_SLOPE = 90  # 频率斜率，用于计算雷达脉冲的频率变化
ADC_SAMPLES = 128  # 每个chirp的ADC采样数量（即数据样本数量）
SAMPLE_RATE = 5000  # 雷达的采样率（单位：kHz），即每秒采集的样本数
RX_GAIN = 30  # 接收增益，表示信号的放大倍数

# 脉冲调制的时间参数
IDLE_TIME = 80  # 闲置时间（单位：微秒），即每个脉冲周期的间隔
RAMP_END_TIME = 44 # 频率扫描的结束时间（单位：微秒）
# NUM_FRAMES 为帧数，0 表示数据会一直流式传输
NUM_FRAMES = 800  # 设置为0时，数据会一直流式传输
# LOOPS_PER_FRAME 每帧中chirp的循环次数，设置为128表示每帧包含128个循环
LOOPS_PER_FRAME = 128  # 每帧的chirp循环次数，一次循环有三个chirp
# PERIODICITY = 100 # 每100ms为周期的设置，注释掉了
# time for one chirp in ms 100ms == 10FPS

# 雷达信号的解析参数
NUM_DOPPLER_BINS = LOOPS_PER_FRAME  # 多普勒 bins 的数量（与循环次数相等）
NUM_RANGE_BINS = ADC_SAMPLES  # 距离 bins 的数量（即 ADC采样数）
NUM_ANGLE_BINS = 64  # 角度 bins 的数量，表示天线的分辨率

# 距离分辨率计算（单位：米）
RANGE_RESOLUTION = (3e8 * SAMPLE_RATE * 1e3) / (2 * FREQ_SLOPE * 1e12 * ADC_SAMPLES)
# 最大距离计算（单位：米）
MAX_RANGE = (300 * SAMPLE_RATE) / (2 * FREQ_SLOPE * 1e3)

# 多普勒分辨率计算（单位：Hz）
DOPPLER_RESOLUTION = 3e8 / (2 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_DOPPLER_BINS * NUM_TX)
# 最大多普勒计算（单位：Hz）
MAX_DOPPLER = 3e8 / (4 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_TX)

# 雷达的位置（硬编码）
MMWAVE_RADAR_LOC = np.array([[0, 0 ,0]])  # 雷达的实际物理位置（x, y, z）
