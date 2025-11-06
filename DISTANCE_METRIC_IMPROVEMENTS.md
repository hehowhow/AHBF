# 距离度量改进说明

## 改进内容

### 1. 新增Wasserstein距离支持

在 `compute_cosine_similarities` 方法中新增了 Wasserstein 距离计算方式，现在支持两种距离度量：
- **cosine**: 余弦相似度（默认）
- **wasserstein**: Wasserstein距离（1维Earth Mover's Distance）

### 2. 计算效率优化

#### 原有问题：
- 使用嵌套循环逐样本计算距离
- 大量的字典查找操作
- 调用 `.item()` 将张量转换为Python标量，破坏了GPU计算的连续性

#### 优化方案：
1. **批量计算**: 将所有样本的历史logits预先构建为一个张量 `[batch_size, num_classes]`
2. **向量化操作**: 使用PyTorch的批量运算替代循环
3. **GPU内存优化**: 避免CPU-GPU数据传输，所有计算保持在GPU上
4. **避免Python循环**: 使用 `torch.stack` 和 `torch.where` 等批量操作

#### 性能提升：
- **时间复杂度**: 从 O(batch_size × num_branches) 次单独计算降低到 O(num_branches) 次批量计算
- **内存效率**: 减少中间变量和临时张量的创建
- **GPU利用率**: 保持计算在GPU上连续进行，避免频繁的CPU-GPU数据传输

### 3. Wasserstein距离实现细节

```python
def compute_wasserstein_distance(self, logit1, logit2):
    # 1. 将logits转换为概率分布
    prob1 = F.softmax(logit1, dim=1)
    prob2 = F.softmax(logit2, dim=1)
    
    # 2. 排序并计算累积分布函数
    sorted_prob1, _ = torch.sort(prob1, dim=1)
    sorted_prob2, _ = torch.sort(prob2, dim=1)
    
    cdf1 = torch.cumsum(sorted_prob1, dim=1)
    cdf2 = torch.cumsum(sorted_prob2, dim=1)
    
    # 3. 计算两个CDF之间的L1距离
    wasserstein_dist = torch.mean(torch.abs(cdf1 - cdf2), dim=1)
    
    return wasserstein_dist
```

这个实现使用了1维Wasserstein距离的快速计算方法，适用于分类任务中的概率分布比较。

### 4. 余弦相似度优化

原实现：
```python
# 逐样本计算，需要多次调用
for i, sample_id in enumerate(sample_ids):
    cos_sim = F.cosine_similarity(
        current_logit.view(1, -1),
        prev_logit.view(1, -1),
        dim=1
    )
    sample_dissimilarities[i] = 1.0 - cos_sim.item()  # CPU-GPU传输
```

优化后：
```python
# 批量计算，一次性处理所有样本
logit_norm = F.normalize(logit, p=2, dim=1)
prev_norm = F.normalize(prev_logits_batch, p=2, dim=1)
cos_sim = torch.sum(logit_norm * prev_norm, dim=1)  # [batch_size]
dissim = 1.0 - cos_sim  # 保持在GPU上
```

## 使用方法

### 命令行参数

```bash
# 使用余弦相似度（默认）
python train_ahbf.py --distance_metric cosine

# 使用Wasserstein距离
python train_ahbf.py --distance_metric wasserstein
```

### 代码中设置

```python
# 创建模型时指定距离度量
model = resnet32(
    num_classes=100,
    num_branches=4,
    aux=2,
    type='conv',
    distance_metric='wasserstein'  # 或 'cosine'
)

# 或者在模型创建后设置
model.distance_metric = 'wasserstein'
```

## 计算性能对比

假设 batch_size=128, num_branches=4, num_classes=100:

### 原实现：
- 嵌套循环: 128 × 4 = 512 次迭代
- 每次迭代包含: 字典查找 + 张量切片 + 相似度计算 + .item() 调用
- 估计时间: ~10-20ms (取决于GPU)

### 优化后：
- 批量操作: 1次历史张量构建 + 4次批量距离计算
- 全GPU计算，无CPU-GPU传输
- 估计时间: ~2-3ms (约5-10倍提速)

## 技术细节

### GPU计算确认
所有计算都在GPU上进行：
- 使用 `device=logitlist[0].device` 确保张量在正确设备上
- 避免使用 `.item()` 将张量转为Python标量
- 使用 `torch.where` 等GPU友好的条件操作

### 内存管理
- 预分配历史logits张量: `prev_logits_batch = torch.zeros(batch_size, num_classes, device=device)`
- 使用boolean mask避免条件分支: `has_history = torch.zeros(batch_size, dtype=torch.bool, device=device)`
- Stack操作实现批量归一化: `dissim_stack = torch.stack(dissimilarities, dim=0)`

## 何时使用哪种距离度量

### 余弦相似度 (Cosine)
- **优点**: 计算快速，关注方向相似性
- **适用**: 当logit的量级可能差异较大时
- **推荐**: 作为默认选择

### Wasserstein距离 (Wasserstein)
- **优点**: 考虑概率分布的几何结构，对outlier更鲁棒
- **适用**: 当需要更细致的分布比较时
- **推荐**: 用于实验对比或特殊场景

## 注意事项

1. **历史记录**: 方法依赖于 `self.prev_ensem_logits` 字典，确保在epoch结束时调用 `update_epoch_history()`
2. **第一个epoch**: 在没有历史记录时自动返回均匀权重
3. **DataParallel**: 如果使用多GPU，需要通过 `model.module` 访问这些属性

## 未来改进方向

1. 支持更多距离度量（如KL散度、JS散度等）
2. 自适应选择距离度量
3. 历史记录的内存管理（限制存储数量）
4. 支持多epoch历史的加权平均


