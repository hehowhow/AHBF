

import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['vgg16', 'vgg19']

#cfg = {
#    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}
class AHBF(nn.Module):
    def __init__(self, inchannel, r=2,  L=64):
        super(AHBF, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel
        self.truinchannle=2*inchannel
        self.conv1 = nn.Conv2d(self.truinchannle, self.inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(self.inchannel)

        self.control_v1 = nn.Linear(self.inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
    def forward(self, x,y,logitx,logity):
        feasc = torch.cat([x, y], dim=1)
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)   ###8/5   1:02 add this line

        feas = self.pool(feasc)
        feas = feas.view(feas.size(0), -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=F.relu(feas)
        feas = F.softmax(feas,dim=1)
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1)
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1)
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit

class VGG(nn.Module):
    def __init__(self, num_classes=10, num_branches=3, aux=3, depth=16, dropout=0.5, distance_metric='cosine'):
        super(VGG, self).__init__()
        self.inplances = 64
        self.aux = aux
        self.num_branches = num_branches
        
        # 添加历史融合输出存储（按样本级别）
        self.use_adaptive_weighting = True  # 控制是否使用自适应加权
        self.distance_metric = distance_metric  # 'cosine' or 'wasserstein'
        self.epoch_count = 0  # 记录当前epoch
        self.prev_ensem_logits = {}  # 存储每个样本的历史融合输出 {sample_id: ensem_logit}
        self.current_epoch_ensem_logits = {}  # 存储当前epoch的融合输出
        
        # 添加历史融合输出存储
        self.register_buffer('prev_ensem_logit', None)
        self.conv1 = nn.Conv2d(3, self.inplances, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplances)
        self.conv2 = nn.Conv2d(self.inplances, self.inplances, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.inplances)
        self.relu = nn.ReLU(inplace=True)            
        self.layer1 = self._make_layers(128, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if depth == 16:
            num_layer = 3
        elif depth == 19:
            num_layer = 4
        
        self.layer2 = self._make_layers(256, num_layer)
        self.layer3 = self._make_layers(512, num_layer)
        self.fixplanes=self.inplances

        for i in range(num_branches):
            setattr(self, 'layer3_'+str(i), self._make_layers(512, num_layer+i*self.aux))
            self.inplances= self.fixplanes

            setattr(self, 'classifier3_'+str(i), nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p = dropout),
            nn.Linear(512, num_classes),
            ))
        for i in range(num_branches-1):
            setattr(self, 'afm_' + str(i), AHBF(512))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def _make_layers(self, input, num_layer):    
        layers=[]
        for i in range(num_layer):
            conv2d = nn.Conv2d(self.inplances, input, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(input), nn.ReLU(inplace=True)]
            self.inplances = input
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
    
    def compute_wasserstein_distance(self, logit1, logit2):
        """
        计算两个logit分布之间的1维Wasserstein距离（批量版本）
        Args:
            logit1: [batch_size, num_classes]
            logit2: [batch_size, num_classes]
        Returns:
            distances: [batch_size] Wasserstein距离
        """
        # 将logits转换为概率分布
        prob1 = F.softmax(logit1, dim=1)
        prob2 = F.softmax(logit2, dim=1)
        
        # 对每个样本排序并计算累积分布函数
        sorted_prob1, _ = torch.sort(prob1, dim=1)
        sorted_prob2, _ = torch.sort(prob2, dim=1)
        
        # 计算累积分布函数
        cdf1 = torch.cumsum(sorted_prob1, dim=1)
        cdf2 = torch.cumsum(sorted_prob2, dim=1)
        
        # Wasserstein距离是两个CDF之间的L1距离
        wasserstein_dist = torch.mean(torch.abs(cdf1 - cdf2), dim=1)
        
        return wasserstein_dist

    def compute_cosine_similarities(self, logitlist, prev_ensem_logit, sample_ids=None):
        """
        计算当前各分支logit与历史融合输出的相异度（样本级别）
        支持余弦相似度和Wasserstein距离两种度量方式
        Args:
            logitlist: 当前各分支的logit列表
            prev_ensem_logit: 上一轮的最后一个融合logit（未使用，保留用于兼容性）
            sample_ids: 样本ID列表，用于索引历史融合输出
        Returns:
            similarities: 归一化后的相异度列表
        """
        if prev_ensem_logit is None or self.epoch_count == 0 or sample_ids is None:
            # 如果是第一个epoch或没有样本ID，返回均匀权重
            batch_size = logitlist[0].size(0)
            return [torch.ones(batch_size, device=logitlist[0].device)] * len(logitlist)

        batch_size = logitlist[0].size(0)
        device = logitlist[0].device
        num_classes = logitlist[0].size(1)

        # 构建历史logits张量，用于批量计算
        # 检查哪些样本有历史记录
        has_history = torch.zeros(batch_size, dtype=torch.bool, device=device)
        prev_logits_batch = torch.zeros(batch_size, num_classes, device=device)
        
        for i, sample_id in enumerate(sample_ids):
            if sample_id in self.prev_ensem_logits:
                has_history[i] = True
                prev_logits_batch[i] = self.prev_ensem_logits[sample_id]
        
        # 如果没有任何历史记录，返回均匀权重
        if not has_history.any():
            return [torch.ones(batch_size, device=device)] * len(logitlist)

        # 批量计算所有分支的相异度
        dissimilarities = []
        for logit in logitlist:
            if self.distance_metric == 'cosine':
                # 批量计算余弦相似度（仅对有历史记录的样本）
                # 归一化logits以计算余弦相似度
                logit_norm = F.normalize(logit, p=2, dim=1)
                prev_norm = F.normalize(prev_logits_batch, p=2, dim=1)
                
                # 计算余弦相似度 (batch_size,)
                cos_sim = torch.sum(logit_norm * prev_norm, dim=1)
                
                # 转换为相异度
                dissim = 1.0 - cos_sim
                
            elif self.distance_metric == 'wasserstein':
                # 批量计算Wasserstein距离
                dissim = self.compute_wasserstein_distance(logit, prev_logits_batch)
                
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
            # 对于没有历史记录的样本，设置相异度为1.0
            dissim = torch.where(has_history, dissim, torch.ones_like(dissim))
            dissimilarities.append(dissim)

        # 批量归一化相异度，使每个样本的权重和为1
        # Stack所有相异度 [num_branches, batch_size]
        dissim_stack = torch.stack(dissimilarities, dim=0)
        
        # 计算每个样本的总相异度 [batch_size]
        total_dissim = dissim_stack.sum(dim=0)
        
        # 归一化 [num_branches, batch_size]
        normalized_dissim = torch.where(
            total_dissim.unsqueeze(0) > 0,
            dissim_stack / total_dissim.unsqueeze(0),
            torch.ones_like(dissim_stack) / len(logitlist)
        )
        
        # 转换回列表格式
        result = [normalized_dissim[i] for i in range(len(logitlist))]
        
        return result

    def update_epoch_history(self):
        """
        在epoch结束时更新历史融合输出（样本级别）
        将当前epoch的融合输出存储为历史参考
        """
        # 将当前epoch的融合输出更新为历史参考
        self.prev_ensem_logits = self.current_epoch_ensem_logits.copy()
        # 清空当前epoch的记录
        self.current_epoch_ensem_logits = {}
        # 更新epoch计数
        self.epoch_count += 1
    
    def forward(self, x, sample_ids=None):
        featurelist = []
        logitlist = []

        # 如果没有提供sample_ids，生成基于batch位置的ID
        if sample_ids is None:
            batch_size = x.size(0)
            # 使用简单的hash来生成样本ID（实际应用中可能需要更复杂的ID生成策略）
            sample_ids = [hash(str(x[i].data_ptr())) for i in range(batch_size)]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_3 = getattr(self, 'layer3_0')(x)  # B x 512 x 1 x 1
        featurelist.append(x_3)

        x_3 = x_3.view(x_3.size(0), -1)  # B x 512
        x_3_1 = getattr(self, 'classifier3_0')(x_3)  # B x num_classes
        logitlist.append(x_3_1)
        
        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_' + str(i))(x)
            featurelist.append(temp)

            temp = temp.view(temp.size(0), -1)
            temp_1 = getattr(self, 'classifier3_' + str(i))(temp)
            logitlist.append(temp_1)
        
        ensem_fea = []
        ensem_logits = []

        # 计算相异度权重（样本级别）
        if self.use_adaptive_weighting:
            dissimilarities = self.compute_cosine_similarities(logitlist, None, sample_ids)
            # dissimilarities已经是tensor列表，每个元素对应一个分支的样本级权重
        else:
            # 如果不使用自适应加权，使用均匀权重
            batch_size = x.size(0)
            dissimilarities = [torch.ones(batch_size, device=x.device, dtype=x.dtype) for _ in range(self.num_branches)]

        for i in range(0, self.num_branches - 1):
            if i == 0:
                # 对第一个融合，使用相异度加权logit（样本级别）
                # 需要将权重扩展为与logit相同的形状
                weight_0 = dissimilarities[i].view(-1, 1)  # [batch_size, 1]
                weight_1 = dissimilarities[i + 1].view(-1, 1)  # [batch_size, 1]
                weighted_logit_0 = logitlist[i] * weight_0
                weighted_logit_1 = logitlist[i + 1] * weight_1
                ensembleff, logit = getattr(self, 'afm_' + str(i))(featurelist[i], featurelist[i + 1],
                                                                   weighted_logit_0, weighted_logit_1)
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                # 对后续融合，使用相异度加权logit（样本级别）
                weight_i_plus_1 = dissimilarities[i + 1].view(-1, 1)  # [batch_size, 1]
                weighted_logit_i_plus_1 = logitlist[i + 1] * weight_i_plus_1
                ensembleff, logit = getattr(self, 'afm_' + str(i))(ensem_fea[i - 1], featurelist[i + 1],
                                                                   ensem_logits[i - 1], weighted_logit_i_plus_1)
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)

        # 存储当前batch的融合输出（样本级别）
        if len(ensem_logits) > 0:
            for j, sample_id in enumerate(sample_ids):
                self.current_epoch_ensem_logits[sample_id] = ensem_logits[-1][j].detach()

        return logitlist, ensem_logits

def vgg16(pretrained=False, path=None, **kwargs):
    model = VGG(depth=16, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
    
def vgg19(pretrained=False, path=None, **kwargs):
    model = VGG(depth=19, **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
