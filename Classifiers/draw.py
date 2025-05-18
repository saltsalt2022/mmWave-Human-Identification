import json

# 加载训练日志
with open('training_log.json', 'r') as f:
    training_log = json.load(f)

# 打印训练集和验证集的损失
print("Training Loss:", training_log['loss'])
print("Validation Loss:", training_log['val_loss'])