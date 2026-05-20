import torch
import torch.nn.functional as F
from torch import nn


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=0.25, alpha=2, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # 计算sigmoid输出
        pt = torch.sigmoid(input)

        # 防止pt等于0或1，避免log(0)的问题
        epsilon = 1e-6
        pt = torch.clamp(pt, min=epsilon, max=1 - epsilon)

        # 计算二元交叉熵损失
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def focal_loss(outputs, targets, gamma=2.0, alpha=0.3):
    """
    计算二分类焦点损失函数

    参数:
    outputs (torch.Tensor): 模型输出的预测概率,shape为(batch_size,)
    targets (torch.Tensor): 真实标签,shape为(batch_size,),取值为0或1
    gamma (float): 调整因子
    alpha (float): 正样本的权重
    """
    outputs = F.sigmoid(outputs)
    epsilon = 1e-7

    p_t = outputs * targets + (1 - outputs) * (1 - targets) + epsilon

    p_t = torch.clamp(p_t, min=epsilon, max=1.0 - epsilon)  # 避免极端值

    loss = -alpha * ((1 - p_t) ** gamma) * torch.log(p_t)
    return loss.mean()


def weighted_cross_entropy_loss(outputs, targets, pos_weight=2, neg_weight=1):
    """
    计算加权交叉熵损失函数


    参数:
    outputs (torch.Tensor): 模型输出的预测概率,shape为(batch_size,)
    targets (torch.Tensor): 真实标签,shape为(batch_size,),取值为0或1
    pos_weight (float): 正样本的权重
    neg_weight (float): 负样本的权重
    """
    outputs = torch.sigmoid(outputs)
    outputs = outputs.to(torch.float32)
    targets = targets.to(torch.float32)
    # 计算交叉熵损失
    bce_loss = F.binary_cross_entropy(outputs, targets, reduction='none')

    # 根据样本类别加权
    weighted_loss = pos_weight * targets * bce_loss + neg_weight * (1 - targets) * bce_loss

    # 计算平均损失
    loss = weighted_loss.mean()

    return loss


class ClassBalancedLoss(nn.Module):
    def __init__(self, beta=0.8, reduction='mean'):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 将输入转换为概率
        probs = torch.sigmoid(inputs)

        # 确保目标张量是长整型，表示类别标签
        targets = targets.long()

        # 计算正负样本的数量
        num_pos = torch.sum(targets == 1).item()
        num_neg = torch.sum(targets == 0).item()

        # 防止除数为零
        num_pos = max(num_pos, 1)
        num_neg = max(num_neg, 1)

        # 计算有效数量
        effective_num_pos = 1.0 - self.beta ** num_pos
        effective_num_neg = 1.0 - self.beta ** num_neg

        # 计算权重
        pos_weight = (1.0 - self.beta) / effective_num_pos
        neg_weight = (1.0 - self.beta) / effective_num_neg

        # 为每个样本创建权重张量
        weights = torch.where(targets == 1, pos_weight, neg_weight).to(inputs.device)

        # 计算加权交叉熵损失
        loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), weight=weights, reduction=self.reduction)

        return loss


def balanced_link_dice_loss(pred_links, true_links, pos_weight=0.3, smooth=1e-8):
    """
    用于链路预测任务的平衡Dice Loss

    参数:
    pred_links (torch.Tensor): 预测的边概率, shape为(E,)
    true_links (torch.Tensor): 真实的边标签, shape为(E,), 取值为0或1
    pos_weight (float): 正样本的权重，用于平衡正负样本
    smooth (float): 防止除零错误的小数值
    """
    # pred_links = torch.sigmoid(pred_links)
    # 计算预测边集和真实边集的交集大小
    intersection = (pred_links * true_links).sum()

    # 计算预测边集和真实边集的并集大小
    # 注意这里使用了pos_weight来加权正样本
    union = (pred_links.sum() + pos_weight * true_links.sum())

    # 计算Dice系数
    dice_coeff = (2. * intersection + smooth) / (union + smooth)

    # Dice Loss定义为1-Dice系数
    dice_loss = 1. - dice_coeff

    return dice_loss