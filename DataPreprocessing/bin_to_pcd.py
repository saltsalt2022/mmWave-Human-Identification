import os
import numpy as np
import open3d as o3d
from tqdm import tqdm


def bin_to_pcd(bin_read_file,pcd_save_file):
    #文件夹路径读取
    bin_pth = bin_read_file
    pcd_pth = pcd_save_file
    #bin文件加载
    files = os.listdir(bin_pth)
    files = [f for f in files if f[-4:] == '.bin']
    #处理文件夹内所有bin文件
    for ic in tqdm(range(len(files)), desc='进度 '):
        f = files[ic]
        #当前处理的bin文件路径生成
        binname = os.path.join(bin_pth, f)
        #转化后保存的pcd文件路径生成
        pcdname = os.path.join(pcd_pth, f[:-4] + '.pcd')
        #bin文件内部点云数据加载-其中3表示存储格式中存储了三列数据根据具体文件修改
        points = np.fromfile(binname, dtype=np.float32).reshape(-1,3)
        #创建pcd点云格式并将numpy格式的点云数据写入保存为pcd文件
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(pcdname,pcd, write_ascii=True)
bin_to_pcd('Data_bin/Train/NO2/lqy/','Data_bin/Train/NO2/lqy/')