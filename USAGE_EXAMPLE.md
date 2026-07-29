# 距离度量使用示例

## 1. 基本使用

### 使用余弦相似度（默认）

```bash
python train_ahbf.py \
    --model resnet32 \
    --dataset CIFAR100 \
    --num_branches 4 \
    --aux 2 \
    --distance_metric cosine \
    --use_adaptive_weighting True
```

### 使用Wasserstein距离

```bash
python train_ahbf.py \
    --model resnet32 \
    --dataset CIFAR100 \
    --num_branches 4 \
    --aux 2 \
    --distance_metric wasserstein \
    --use_adaptive_weighting True
```

## 2. 完整训练命令示例

### CIFAR-100 训练（余弦相似度）

```bash
python train_ahbf.py \
    --model resnet32 \
    --dataset CIFAR100 \
    --root ./Data \
    --num_epochs 300 \
    --batch_size 128 \
    --lr 0.1 \
    --schedule 150 225 \
    --num_branches 4 \
    --aux 2 \
    --kd_T 3.0 \
    --kd_weight 1.5 \
    --lambda1 1.0 \
    --lambda2 1.0 \
    --att_type conv \
    --distance_metric cosine \
    --use_adaptive_weighting True \
    --gpu_id 0 \
    --notes "cosine_baseline"
```

### CIFAR-100 训练（Wasserstein距离）

```bash
python train_ahbf.py \
    --model resnet32 \
    --dataset CIFAR100 \
    --root ./Data \
    --num_epochs 300 \
    --batch_size 128 \
    --lr 0.1 \
    --schedule 150 225 \
    --num_branches 4 \
    --aux 2 \
    --kd_T 3.0 \
    --kd_weight 1.5 \
    --lambda1 1.0 \
    --lambda2 1.0 \
    --att_type conv \
    --distance_metric wasserstein \
    --use_adaptive_weighting True \
    --gpu_id 0 \
    --notes "wasserstein_test"
```

## 3. 对比实验

### 运行对比实验脚本

创建 `compare_metrics.sh`:

```bash
#!/bin/bash

# CIFAR-100 ResNet32 with different distance metrics

# Baseline: Cosine similarity
python train_ahbf.py \
    --model resnet32 \
    --dataset CIFAR100 \
    --num_branches 4 \
    --aux 2 \
    --distance_metric cosine \
    --gpu_id 0 \
    --notes "exp_cosine" &

# Experiment: Wasserstein distance
python train_ahbf.py \
    --model resnet32 \
    --dataset CIFAR100 \
    --num_branches 4 \
    --aux 2 \
    --distance_metric wasserstein \
    --gpu_id 1 \
    --notes "exp_wasserstein" &

wait
echo "All experiments completed!"
```

运行:
```bash
chmod +x compare_metrics.sh
./compare_metrics.sh
```

## 4. 在代码中使用

### 方式1: 创建模型时指定

```python
import torch
from models.model_backbone.resnet_ahbf import resnet32

# 使用余弦相似度
model_cosine = resnet32(
    num_classes=100,
    num_branches=4,
    aux=2,
    type='conv',
    distance_metric='cosine'
)

# 使用Wasserstein距离
model_wasserstein = resnet32(
    num_classes=100,
    num_branches=4,
    aux=2,
    type='conv',
    distance_metric='wasserstein'
)
```

### 方式2: 创建后修改

```python
import torch
from models.model_backbone.resnet_ahbf import resnet32

# 创建模型（默认使用cosine）
model = resnet32(num_classes=100, num_branches=4, aux=2, type='conv')

# 切换到Wasserstein距离
model.distance_metric = 'wasserstein'

# 启用自适应加权
model.use_adaptive_weighting = True
```

## 5. 测试和验证

### 运行测试脚本

```bash
python test_distance_metrics.py
```

这将测试:
- 两种距离度量的正确性
- 计算性能
- 权重归一化
- Wasserstein距离的数学性质

### 预期输出示例

```
================================================================================
距离度量测试
================================================================================

使用设备: cuda

============================================================
测试距离度量: cosine
============================================================

第一次前向传播（无历史记录）...
  - 耗时: 15.23 ms
  - 分支输出数量: 4
  - 融合输出数量: 3
  - 输出形状: torch.Size([128, 100])

第二次前向传播（有历史记录）...
  - 耗时: 16.45 ms
  - 输出形状: torch.Size([128, 100])

测试 compute_cosine_similarities 性能...
  - 平均耗时: 2.145 ms (100次迭代)
  - 返回值数量: 4
  - 权重形状: torch.Size([128])
  - 权重和: 1.0000 (应该接近1.0)
  - 样本0的权重: ['0.252', '0.248', '0.251', '0.249'], 和=1.0000
```

## 6. 性能监控

### 使用WandB监控

训练过程中，可以在WandB中查看:
- `trainloss`: 总训练损失
- `train_acc_target`: 目标分支准确率
- `train_acc_afm0/1/2`: 各融合层准确率
- 训练时间对比

### 查看训练日志

```bash
# 查看特定实验的日志
tail -f CIFAR100/300/resnet32B4T3.0A2Nexp_cosine/train.log

# 比较两个实验的最佳准确率
grep "best_acc" results1*.txt
```

## 7. 注意事项

### GPU内存

Wasserstein距离计算需要额外的排序操作，可能会略微增加内存使用:

```python
# 如果遇到OOM，可以尝试:
# 1. 减小batch size
python train_ahbf.py --batch_size 64 --distance_metric wasserstein

# 2. 减少分支数量
python train_ahbf.py --num_branches 3 --distance_metric wasserstein
```

### 多GPU训练

使用DataParallel时，需要通过module访问属性:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    # 访问属性时需要使用.module
    model.module.distance_metric = 'wasserstein'
    model.module.use_adaptive_weighting = True
else:
    model.distance_metric = 'wasserstein'
    model.use_adaptive_weighting = True
```

### 禁用自适应加权

如果想要禁用自适应加权（使用均匀权重）:

```bash
python train_ahbf.py --use_adaptive_weighting False
```

## 8. 实验建议

### 初次使用

1. 先用小数据集和短训练周期验证:
```bash
python train_ahbf.py \
    --dataset CIFAR10 \
    --num_epochs 10 \
    --distance_metric cosine
```

2. 比较两种距离度量的训练曲线

3. 选择效果更好的方法进行完整训练

### 超参数调优

不同距离度量可能需要不同的超参数:

```bash
# Cosine similarity (原始设置)
python train_ahbf.py --distance_metric cosine --kd_weight 1.5

# Wasserstein distance (可能需要调整)
python train_ahbf.py --distance_metric wasserstein --kd_weight 1.2
```

### 消融实验

```bash
# 1. 无自适应加权（baseline）
python train_ahbf.py --use_adaptive_weighting False

# 2. 余弦相似度加权
python train_ahbf.py --distance_metric cosine --use_adaptive_weighting True

# 3. Wasserstein距离加权
python train_ahbf.py --distance_metric wasserstein --use_adaptive_weighting True
```

## 9. 故障排除

### 问题1: 训练速度变慢

**原因**: Wasserstein距离计算比余弦相似度稍慢

**解决方案**:
- 使用更大的batch size
- 减少num_branches
- 或切换回cosine

### 问题2: 准确率没有提升

**原因**: 可能需要调整超参数

**解决方案**:
- 调整kd_weight
- 调整lambda1和lambda2
- 尝试不同的attention类型 (--att_type)

### 问题3: 权重不归一化

**检查**:
```python
# 在forward中打印权重
similarities = self.compute_cosine_similarities(logitlist, None, sample_ids)
print(f"Weight sum: {sum([s.sum().item() for s in similarities]) / batch_size}")
# 应该输出接近1.0
```

## 10. 引用和参考

如果使用了这些改进，请引用原始AHBF论文，并说明使用了优化的距离度量实现。

```
改进内容:
- 批量化距离计算 (5-10x加速)
- 支持Wasserstein距离
- GPU内存优化
```


