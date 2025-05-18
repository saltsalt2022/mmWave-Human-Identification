"""
Time-Distributed CNN + Bidirectional LSTM on VOXELS

- extract_path 是接收处理好的数据格式文件的路径
- checkpoint_model_path 是训练时保存模型的路径
"""
"""
代码说明：
修改原来对动作类别的分类，改为人名
1. 读取体素化点云数据，合并所有类别，标签做one-hot编码。
2. 划分训练集和验证集。
3. 优先加载已保存模型（断点续训），否则新建模型。
4. 构建 TimeDistributed 3D CNN + BiLSTM 网络，提取空间和时序特征。
5. 训练过程中保存最优模型，并将训练日志保存为json文件。
"""
import glob
import os
import numpy as np
from numpy.random import seed
import tensorflow
import json

# 设置随机种子，保证实验可复现
rand_seed = 1
seed(rand_seed)
tensorflow.random.set_seed(rand_seed)

# 列出所有的 GPU 设备，并设置显存自适应增长
physical_devices = tensorflow.config.list_physical_devices('GPU')
for device in physical_devices:
    tensorflow.config.experimental.set_memory_growth(device, True)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, Conv3D, MaxPooling2D, MaxPooling3D,
    LSTM, Dense, Dropout, Flatten, Bidirectional,
    TimeDistributed, Activation, BatchNormalization,
    Permute, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ================== 路径与类别设置 ==================
extract_path = 'deal_Data_bin/Train/'         # 处理好的数据路径
checkpoint_model_path = "./model.keras"       # 模型保存路径
sub_dirs = ['lqy', 'hdt', 'lxb', 'lhc']      # 动作类别

# ================== One-hot 编码函数 ==================
def one_hot_encoding(y_data, sub_dirs, categories=4):
    """
    将文本标签转换为 one-hot 编码
    """
    Mapping = {name: idx for idx, name in enumerate(sub_dirs)}
    y_features = [Mapping[y] for y in y_data]
    y_features = np.array(y_features).reshape(-1, 1)
    return to_categorical(y_features, num_classes=categories)

# ================== 模型构建函数 ==================
def full_3D_model(input_x, input_y, reg=0, num_feat_map=16, summary=False):
    """
    构建 TimeDistributed 3D CNN + 双向 LSTM 模型
    """
    print('building the model ... ')
    model = Sequential()

    # 第一组 TimeDistributed 3D CNN
    model.add(TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv1a",
                                     input_shape=(10, 32, 32, 1), padding="same", activation="relu")))
    model.add(TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv1b",
                                     padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool1")))

    # 第二组 TimeDistributed 3D CNN
    model.add(TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv2a", padding="same", activation="relu")))
    model.add(TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv2b", padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                           data_format="channels_first", name="pool2")))

    # 第三组 TimeDistributed 3D CNN
    model.add(TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv3a", padding="same", activation="relu")))
    model.add(TimeDistributed(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), name="conv3b", padding="same", activation="relu")))
    model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid",
                                           data_format="channels_first", name="pool3")))

    model.add(TimeDistributed(Flatten()))  # 展平3D特征
    model.add(Dropout(0.5))

    # 双向 LSTM 提取时序特征
    model.add(Bidirectional(LSTM(16, return_sequences=False)))
    model.add(Dropout(0.3))

    # 输出层：分类
    model.add(Dense(input_y.shape[1], activation='softmax', name='output'))

    if summary:
        model.summary()
    return model

# ================== 数据加载与处理 ==================
# 读取各类别数据并合并
Data_path = extract_path + 'lqy'
data = np.load(Data_path + '.npz')
train_data = data['arr_0']
train_label = data['arr_1']
del data

for action in ['hdt', 'lxb', 'lhc']:
    Data_path = extract_path + action
    data = np.load(Data_path + '.npz')
    train_data = np.concatenate((train_data, data['arr_0']), axis=0)
    train_label = np.concatenate((train_label, data['arr_1']), axis=0)
    del data

# 标签 one-hot 编码
train_label = one_hot_encoding(train_label, sub_dirs, categories=4)

# 重组数据格式：(N, T, D, H, W, 1)
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1],
                                train_data.shape[2], train_data.shape[3],
                                train_data.shape[4], 1)

print('Training Data Shape is:')
print(train_data.shape, train_label.shape)

# ================== 划分训练集和验证集 ==================
X_train, X_val, y_train, y_val = train_test_split(
    train_data, train_label, test_size=0.20, random_state=1)
del train_data, train_label

# ================== 构建或加载模型 ==================
if os.path.exists(checkpoint_model_path):
    print("Loading existing model for continued training...")
    model = load_model(checkpoint_model_path)
else:
    print("Building new model...")
    model = full_3D_model(X_train, y_train)
    print("Model building is completed")

# 配置优化器
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# 模型保存回调器
checkpoint = ModelCheckpoint(
    checkpoint_model_path, monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]

# ================== 训练模型 ==================
learning_hist = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=30,            # 总训练轮数
    initial_epoch=13,     # 从第13轮继续训练
    verbose=1,
    shuffle=True,
    validation_data=(X_val, y_val),
    callbacks=callbacks_list
)

# ================== 保存训练日志 ==================
with open('training_log.json', 'w') as f:
    json.dump(learning_hist.history, f)

