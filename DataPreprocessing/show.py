import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# 点云文件夹路径
pointcloud_folder = './bin_to_txt/NO2'  # 替换为存储点云文件的文件夹路径
pointcloud_files = sorted([f for f in os.listdir(pointcloud_folder) if f.endswith('.txt')])  # 获取所有点云文件

# 加载所有点云数据
pointclouds = []
for file in pointcloud_files:
    filepath = os.path.join(pointcloud_folder, file)
    pointcloud = np.loadtxt(filepath, skiprows=1)  # 跳过第一行
    pointclouds.append(pointcloud)

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化点云图
scatter = ax.scatter([], [], [], c=[], cmap='hot', s=5)

# 设置坐标轴范围（根据点云数据范围调整）
all_points = np.vstack(pointclouds)  # 合并所有点云数据
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 动画更新函数
def update(frame):
    ax.clear()  # 清除上一帧内容
    pointcloud = pointclouds[frame]
    x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
    velocity = pointcloud[:, 3] if pointcloud.shape[1] > 3 else z  # 使用第 4 列（如速度）作为颜色
    scatter = ax.scatter(x, y, z, c=velocity, cmap='hot', s=5)  # 使用速度值调整颜色强烈程度
    ax.set_title(f'Frame {frame + 1}/{len(pointclouds)}: {pointcloud_files[frame]}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    return scatter,

# 创建动画
ani = FuncAnimation(fig, update, frames=len(pointclouds), interval=20, repeat=True)

# 显示动画
plt.show()