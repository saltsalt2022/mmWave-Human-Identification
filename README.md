# mmWave Radar-Based Human Identification

本项目基于 [RadHAR: Human Activity Recognition from Point Clouds Generated through a Millimeter-wave Radar](https://dl.acm.org/citation.cfm?id=3356768) 的代码进行修改和扩展，旨在实现毫米波雷达点云数据的人体身份识别。

## 原始项目引用

我们对原始项目的作者表示由衷的感谢，原始项目的代码和数据集为本项目提供了重要的基础。原始项目的详细信息如下：

```
@inproceedings{singh2019radhar,
  title={RadHAR: Human Activity Recognition from Point Clouds Generated through a Millimeter-wave Radar},
  author={Singh, Akash Deep and Sandha, Sandeep Singh and Garcia, Luis and Srivastava, Mani},
  booktitle={Proceedings of the 3rd ACM Workshop on Millimeter-wave Networks and Sensing Systems},
  pages={51--56},
  year={2019},
  organization={ACM}
}
```

原始项目的代码使用 [BSD-3-Clause License](LICENSE)，我们保留了原始项目的许可证文件。

## 修改内容

本项目在原始项目的基础上进行了以下修改和扩展：

1. **任务转换**：将原始的动作识别任务改为人体身份识别任务。
2. **数据处理改进**：新增了直接从 `.bin` 文件转换为点云数据，再将点云数据转换为体素表示的代码，具体实现见 [`DataPreprocessing/bin_to_pointcloud.py`](DataPreprocessing/bin_to_pointcloud.py) 和 [`DataPreprocessing/voxels bin.py`](DataPreprocessing/voxels%20bin.py)。
3. **环境配置**：新增了基于 Docker 的环境配置，便于快速部署和训练模型。

## 项目用途

本项目可用于基于毫米波雷达点云数据的人体身份识别任务，适用于安防、智能家居等场景。

## 环境配置

### 依赖安装

请按照以下步骤配置环境：

1. **安装 TensorFlow 支持的 CUDA 和 cuDNN**  
   请确保您的系统支持 TensorFlow 所需的 CUDA 和 cuDNN 版本。若不符合其中任何一个，建议使用 WSL + Docker + 预装了适配的 cuDNN 和 CUDA 的 TensorFlow 镜像，避免兼容性问题。

2. **Windows 11 在 Docker 中使用 GPU 的教程**  
   - **WSL 代理网络配置（可选，后面实际没用上）**  
     使用 Clash for Windows 勾选“允许局域网代理”，启用 TUN 模式和系统代理。  
     电脑主机也需要配置代理，设置路径：`控制面板\网络和 Internet\Internet选项\连接\局域网设置`，勾选“代理服务器”中的两个选项，并设置地址为 `http://127.0.0.1:7890`。  
     配置 Hyper-V 防火墙：运行 PowerShell，输入以下命令：  
     ```powershell
     Set-NetFirewallHyperVVMSetting -Name ‘{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}’ -DefaultInboundAction Allow
     ```
     然后在开始栏输入wsl settings并进入，在 WSL 设置中选择联网为mirror即可。
    
      官网教程：[Accessing network applications with WSL](https://learn.microsoft.com/en-us/windows/wsl/networking).
   - **安装 NVIDIA Container Toolkit**  
     NVIDIA 提供国内服务器，无需代理即可下载。参考官方文档：[NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

     
### Docker 环境配置

 1. **拉取 TensorFlow GPU 镜像**  
    使用以下命令拉取预装了适配 CUDA 和 cuDNN 的 TensorFlow 镜像：
    ```bash
    docker pull tensorflow/tensorflow:2.13.0-gpu
    ```

 2. **启动容器并挂载工作目录**  
    ```bash
    docker run --gpus all -it --rm \
      -v <你要挂载的目录>:/workspace \
      tensorflow/tensorflow:2.13.0-gpu bash
    ```

 3. **安装 Python 依赖**  
    在容器中运行以下命令安装依赖：
    ```bash
    pip install scikit-learn tqdm
    pip install open3d
    若网速不好
    pip install open3d -i https://pypi.doubanio.com/simple
    若报错部分库无法卸载重装，It is a distutils installed project 
    找到库所在位置，然后rm rf清除对应库
    ```

 4. **保存容器为镜像**  
      退出容器后，保存当前容器为新的镜像：
      ```bash
      docker ps -a  # 找到容器 ID
      docker commit <CONTAINER_ID> mmwave-human-id
      ```




## 数据处理流程

1. **测试, 从 `.bin` 文件生成点云数据 `.txt`**：
   使用 [`DataPreprocessing/bin_to_pointcloud.py`](DataPreprocessing/bin_to_pointcloud.py) 将原始 `.bin` 文件转换为点云数据,并保存为`.txt`，再使用[`DataPreprocessing/show.py`](DataPreprocessing/show.py)可视化。
   ```bash
   python DataPreprocessing/bin_to_pointcloud.py <bin_file_path> <total_frames>
   ```

2. **点云数据转换为体素表示,从 `.bin` 文件生成训练数据 `your tag.npz`**：
   使用 [`DataPreprocessing/voxels bin.py`](DataPreprocessing/voxels%20bin.py) 将点云数据转换为体素表示的训练数据。
   ```bash
   python DataPreprocessing/voxels_bin.py
   ```

3. **开始训练**：
   使用 [`Classifiers/TD_CNN_LSTM_new.py`](Classifiers/TD_CNN_LSTM_new.py) 开始训练模型。

## 许可证

本项目基于 [BSD-3-Clause License](LICENSE) 许可证发布。根据原始项目的许可证要求，我们保留了原始项目的许可证文件，并在此基础上添加了我们的修改内容。

## 致谢

特别感谢原始项目的作者 Akash Deep Singh, Sandeep Singh Sandha, Luis Garcia 和 Mani Srivastava 提供的基础代码。

如果您在研究或项目中使用了本代码，请引用原始论文目。

如果您觉得本项目对您有帮助，请为本项目点亮一颗星星（⭐）！感谢！