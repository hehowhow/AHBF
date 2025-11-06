

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet','ResNet_v2','resnet18', 'resnet32','resnet32_d','resnet32_p','resnet34', 'resnet50','resnet110']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class NonLocalBlockND(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlockND, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class AHBF(nn.Module):
    def __init__(self, inchannel,  r=2,  L=64):
        super(AHBF, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Linear(inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier=nn.Linear(64 , 100)
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
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit
class AHBF_cbam(nn.Module):
    def __init__(self, inchannel,  r=2,  L=64):
        super(AHBF_cbam, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Linear(inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)


        self.cbam=CBAM(64)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier=nn.Linear(64 , 100)
    def forward(self, x,y,logitx,logity):

        feasc = torch.cat([x, y], dim=1)
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)   ###8/5   1:02 add this line
        feasc=self.cbam(feasc)
        feas = self.pool(feasc)
        feas = feas.view(feas.size(0), -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=F.relu(feas)
        feas = F.softmax(feas,dim=1)
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit
class AHBF_non(nn.Module):
    def __init__(self, inchannel,  r=2,  L=64):
        super(AHBF_non, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Linear(inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)


        self.non = NonLocalBlockND(64)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier=nn.Linear(64 , 100)
    def forward(self, x,y,logitx,logity):

        feasc = torch.cat([x, y], dim=1)
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)   ###8/5   1:02 add this line
        feasc=self.non(feasc)
        feas = self.pool(feasc)
        feas = feas.view(feas.size(0), -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=F.relu(feas)
        feas = F.softmax(feas,dim=1)
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit
class AHBF_se(nn.Module):
    def __init__(self, inchannel,  r=2,  L=64):
        super(AHBF_se, self).__init__()
        d = max(int(inchannel*r), L)
        self.inchannel = inchannel

        self.conv1 = nn.Conv2d(2*inchannel, inchannel,1,stride=1)   ###8/5 1:02 kernel3 ---》1
        self.bn1 = nn.BatchNorm2d(inchannel)

        self.control_v1 = nn.Linear(inchannel, 2)

        self.bn_v1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)


        self.se=SE_Block(64)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier=nn.Linear(64 , 100)
    def forward(self, x,y,logitx,logity):

        feasc = torch.cat([x, y], dim=1)
        feasc=self.conv1(feasc)
        feasc=self.bn1(feasc)   ###8/5   1:02 add this line
        feasc=self.se(feasc)
        feas = self.pool(feasc)
        feas = feas.view(feas.size(0), -1)

        feas=self.control_v1(feas)
        feas=self.bn_v1(feas)
        feas=F.relu(feas)
        feas = F.softmax(feas,dim=1)
        x_c_1=feas[:,0].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit = feas[:, 0].view(-1, 1).repeat(1, logitx.size(1)) * logitx


        x_c_2=feas[:,1].repeat(logitx.size()[1], 1).transpose(0,1).contiguous()
        logit += feas[:, 1].view(-1, 1).repeat(1, logity.size(1)) * logity

        logit=x_c_1*logitx+x_c_2*logity    #


        return feasc,logit


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_branches=3, aux=0, type='conv',zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, 
                 distance_metric='cosine'):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_branches = num_branches
        # 添加历史融合输出存储（按样本级别）
        self.use_adaptive_weighting = True  # 控制是否使用自适应加权
        self.distance_metric = distance_metric  # 'cosine' or 'wasserstein'
        self.epoch_count = 0  # 记录当前epoch
        self.prev_ensem_logits = {}  # 存储每个样本的历史融合输出 {sample_id: ensem_logit}
        self.current_epoch_ensem_logits = {}  # 存储当前epoch的融合输出

        # 添加历史融合输出存储
        self.register_buffer('prev_ensem_logit', None)
        self.use_adaptive_weighting = True  # 控制是否使用自适应加权

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        fix_inplanes = self.inplanes  # 32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for i in range(num_branches):
            setattr(self, 'layer3_' + str(i), \
            self._make_layer(block, 64, layers[2] + i * aux, stride=2))
            self.inplanes = fix_inplanes  ##reuse self.inplanes
            setattr(self, 'classifier3_' + str(i), \
                    nn.Linear(64 * block.expansion, num_classes))
        if type == 'conv':
            for i in range(num_branches - 1):
                setattr(self, 'afm_' + str(i), AHBF(64 * block.expansion))
        elif type == 'se':
            for i in range(num_branches - 1):
                setattr(self, 'afm_' + str(i), AHBF_se(64 * block.expansion))
        elif type == 'nonlocal':
            for i in range(num_branches - 1):
                setattr(self, 'afm_' + str(i), AHBF_non(64 * block.expansion))
        elif type == 'cbam':
            for i in range(num_branches - 1):
                setattr(self, 'afm_' + str(i), AHBF_cbam(64 * block.expansion))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

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
        # print('*'*50)
        # score_list=[]
        featurelist = []
        featurelist1 = []
        logitlist = []
        # print(x.shape,'xxxx')

        # 如果没有提供sample_ids，生成基于batch位置的ID
        if sample_ids is None:
            batch_size = x.size(0)
            # 使用简单的hash来生成样本ID（实际应用中可能需要更复杂的ID生成策略）
            sample_ids = [hash(str(x[i].data_ptr())) for i in range(batch_size)]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32

        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x_3 = getattr(self, 'layer3_0')(x)  # B x 64 x 8 x 8
        # print(x_3.shape,'x_3')
        featurelist.append(x_3)
        x_3 = self.avgpool(x_3)  # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)  # B x 64
        # featurelist1.append(x_3)

        x_3_1 = getattr(self, 'classifier3_0')(x_3)  # B x num_classes
        logitlist.append(x_3_1)

        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_' + str(i))(x)
            # print(temp.shape,'temp')
            featurelist.append(temp)

            temp = self.avgpool(temp)  # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)
            # featurelist1.append(temp)
            temp_out = getattr(self, 'classifier3_' + str(i))(temp)
            logitlist.append(temp_out)

        ensem_fea = []
        ensem_logits = []

        # 计算相异度权重
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
                # score_list.append(sco_list)
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                # 对后续融合，使用相异度加权logit（样本级别）
                weight_i_plus_1 = dissimilarities[i + 1].view(-1, 1)  # [batch_size, 1]
                weighted_logit_i_plus_1 = logitlist[i + 1] * weight_i_plus_1
                ensembleff, logit = getattr(self, 'afm_' + str(i))(ensem_fea[i - 1], featurelist[i + 1],
                                                                   ensem_logits[i - 1], weighted_logit_i_plus_1)
                # score_list.append(sco_list)
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)

            # 存储当前batch的融合输出（样本级别）
        if len(ensem_logits) > 0:
            for j, sample_id in enumerate(sample_ids):
                self.current_epoch_ensem_logits[sample_id] = ensem_logits[-1][j].detach()

        return logitlist, ensem_logits

        # return logitlist, ensem_logits, featurelist



class ResNet_ceonly(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_branches=3, aux=0, type='conv', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet_ceonly, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_branches = num_branches

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        fix_inplanes = self.inplanes  # 32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for i in range(num_branches):
            setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2] + i * aux, stride=2))
            self.inplanes = fix_inplanes  ##reuse self.inplanes
            setattr(self, 'classifier3_' + str(i), nn.Linear(64 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        featurelist = []
        featurelist1 = []
        logitlist = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32

        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x_3 = getattr(self, 'layer3_0')(x)  # B x 64 x 8 x 8
        featurelist.append(x_3)
        x_3 = self.avgpool(x_3)  # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)  # B x 64
        featurelist1.append(x_3)

        x_3_1 = getattr(self, 'classifier3_0')(x_3)  # B x num_classes
        logitlist.append(x_3_1)

        for i in range(1, self.num_branches):
            temp = getattr(self, 'layer3_' + str(i))(x)
            featurelist.append(temp)

            temp = self.avgpool(temp)  # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)
            featurelist1.append(temp)
            temp_out = getattr(self, 'classifier3_' + str(i))(temp)
            logitlist.append(temp_out)


        return logitlist


class ResNet_reduce(nn.Module):
    def __init__(self, block, layers, num_classes=10, num_branches = 3,aux=0,   zero_init_residual=False,
        groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet_reduce, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_branches = num_branches

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        fix_inplanes=self.inplanes    # 32
        for i in range(num_branches):
            if i!=num_branches-1:
                setattr(self, 'layer2_' + str(i), self._make_layer(block, 32, layers[1]-aux*(i>1), stride=2))
                setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2]-aux*(i>0), stride=2))
                self.inplanes = fix_inplanes  ##reuse self.inplanes
                setattr(self, 'classifier3_' +str(i), nn.Linear(64 * block.expansion, num_classes))
            else:
                setattr(self, 'layer2_' + str(i), self._make_layer(block, 32, layers[1]-aux, stride=2))
                setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2]-aux*2, stride=2))
                self.inplanes = fix_inplanes  ##reuse self.inplanes
                setattr(self, 'classifier3_' +str(i), nn.Linear(64 * block.expansion, num_classes))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(num_branches-1):
            setattr(self, 'afm_' + str(i), AHBF(64*block.expansion))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        featurelist=[]
        logitlist=[]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)            # B x 16 x 32 x 32

        x = self.layer1(x)          # B x 16 x 32 x 32
        x_2 = getattr(self,'layer2_0')(x)   # B x 64 x 8 x 8
        x_3 = getattr(self,'layer3_0')(x_2)   # B x 64 x 8 x 8
        featurelist.append(x_3)
        x_3 = self.avgpool(x_3)             # B x 64 x 1 x 1
        x_3 = x_3.view(x_3.size(0), -1)     # B x 64
        x_3_1 = getattr(self, 'classifier3_0')(x_3)     # B x num_classes
        logitlist.append(x_3_1)
        for i in range(1, self.num_branches ):
            temp = getattr(self, 'layer2_'+str(i))(x)
            temp = getattr(self, 'layer3_'+str(i))(temp)
            featurelist.append(temp)

            temp = self.avgpool(temp)       # B x 64 x 1 x 1
            temp = temp.view(temp.size(0), -1)
            temp_out = getattr(self, 'classifier3_' + str(i ))(temp)
            logitlist.append(temp_out)
        logitlist=logitlist[::-1]
        featurelist=featurelist[::-1]
        ensem_fea = []
        ensem_logits = []

        for i in range(0,self.num_branches-1):
            if i==0:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(featurelist[i],featurelist[i+1],logitlist[i],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(ensem_fea[i-1],featurelist[i+1],ensem_logits[i-1],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)



        return logitlist, ensem_logits

class ResNet_v2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_branches = 3,aux=0,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, KD = False):
        super(ResNet_v2, self).__init__()
        self.aux=aux
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_branches=num_branches
        self.KD = KD
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        fix_inplanes=self.inplanes    # 32
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        for i in range(num_branches):
            if i%2==0:
                aux_a=i//2
                aux_b=i//2  ###add auxiliary layers monotonously stage by stage   for optimization stability add lower layer first

            else:
                aux_a=i//2+1
                aux_b=i//2  ###add auxiliary layers monotonously stage by stage


            setattr(self, 'layer3_' + str(i), self._make_layer(block, 256, layers[2]+self.aux*(aux_a), stride=2,
                                       dilate=replace_stride_with_dilation[1]))
            setattr(self, 'layer4_' + str(i), self._make_layer(block, 512, layers[3]+self.aux*(aux_b), stride=2,
                                       dilate=replace_stride_with_dilation[2]))
            self.inplanes = fix_inplanes  ##reuse self.inplanes
            setattr(self, 'classifier4_' +str(i), nn.Linear(512 * block.expansion, num_classes))

        for i in range(num_branches-1):
            setattr(self, 'afm_' + str(i), AHBF(512*block.expansion))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        featurelist=[]
        logitlist=[]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = getattr(self,'layer3_0')(x)   # B x 64 x 8 x 8
        x_4 = getattr(self,'layer4_0')(x_3)
        featurelist.append(x_4)
        x_4 = self.avgpool(x_4)             # B x 64 x 1 x 1
        x_4 = x_4.view(x_4.size(0), -1)     # B x 64
        x_4_1 = getattr(self, 'classifier4_0')(x_4)     # B x num_classes
        logitlist.append(x_4_1)
        for i in range(1, self.num_branches ):
            x_3 = getattr(self, 'layer3_'+str(i))(x)  # B x 64 x 8 x 8
            x_4 = getattr(self, 'layer4_'+str(i))(x_3)
            featurelist.append(x_4)
            x_4 = self.avgpool(x_4)  # B x 64 x 1 x 1
            x_4 = x_4.view(x_4.size(0), -1)  # B x 64
            x_4_1 = getattr(self, 'classifier4_'+str(i))(x_4)  # B x num_classes
            logitlist.append(x_4_1)


        ensem_fea = []
        ensem_logits = []

        for i in range(0,self.num_branches-1):
            if i==0:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(featurelist[i],featurelist[i+1],logitlist[i],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)
            else:
                ensembleff,logit=getattr(self, 'afm_'+str(i))(ensem_fea[i-1],featurelist[i+1],ensem_logits[i-1],logitlist[i+1])
                ensem_logits.append(logit)
                ensem_fea.append(ensembleff)

        return logitlist, ensem_logits

def resnet32(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def resnet32_p(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet_ceonly(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model
def resnet32_d(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-32 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet_reduce(BasicBlock, [5, 5, 5], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model

def resnet110(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(Bottleneck, [12, 12, 12], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model



def resnet18(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet_v2(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def resnet34(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet_v2(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model


def resnet50(pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """

    model = ResNet_v2(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict((torch.load(path))['state_dict'])
    return model





