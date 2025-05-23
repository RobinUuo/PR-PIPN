import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, Ns=1.0, category=4):
        super(Network, self).__init__()

        self.fc1_1 = nn.Conv1d(in_channels=3, out_channels=int(64 * Ns), kernel_size=1)
        self.bn1_1 = nn.BatchNorm1d(int(64 * Ns))
        self.fc1_2 = nn.Conv1d(in_channels=int(64 * Ns), out_channels=int(64 * Ns), kernel_size=1)
        self.bn1_2 = nn.BatchNorm1d(int(64 * Ns))

        self.fc2_1 = nn.Conv1d(in_channels=int(64 * Ns), out_channels=int(64 * Ns), kernel_size=1)
        self.bn2_1 = nn.BatchNorm1d(int(64 * Ns))
        self.fc2_2 = nn.Conv1d(in_channels=int(64 * Ns), out_channels=int(128 * Ns), kernel_size=1)
        self.bn2_2 = nn.BatchNorm1d(int(128 * Ns))
        self.fc2_3 = nn.Conv1d(in_channels=int(128 * Ns), out_channels=int(1024 * Ns), kernel_size=1)
        self.bn2_3 = nn.BatchNorm1d(int(1024 * Ns))

        self.fc3_1 = nn.Conv1d(in_channels=int(1088 * Ns), out_channels=int(512 * Ns), kernel_size=1)
        self.bn3_1 = nn.BatchNorm1d(int(512 * Ns))
        self.fc3_2 = nn.Conv1d(in_channels=int(512 * Ns), out_channels=int(256 * Ns), kernel_size=1)
        self.bn3_2 = nn.BatchNorm1d(int(256 * Ns))
        self.fc3_3 = nn.Conv1d(in_channels=int(256 * Ns), out_channels=int(128 * Ns), kernel_size=1)
        self.bn3_3 = nn.BatchNorm1d(int(128 * Ns))
        self.fc3_4 = nn.Conv1d(in_channels=int(128 * Ns), out_channels=int(128 * Ns), kernel_size=1)
        self.bn3_4 = nn.BatchNorm1d(int(128 * Ns))

        # 输出层
        self.prediction = nn.Conv1d(in_channels=int(128 * Ns), out_channels=category, kernel_size=1)

        # 初始化权重
        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        g = F.tanh(self.bn1_1(self.fc1_1(x.unsqueeze(2))))
        g = F.tanh(self.bn1_2(self.fc1_2(g)))
        seg_part1 = g

        g = F.tanh(self.bn2_1(self.fc2_1(g)))
        g = F.tanh(self.bn2_2(self.fc2_2(g)))
        g = F.tanh(self.bn2_3(self.fc2_3(g)))

        # 使用动态最大池化
        global_feature, _ = torch.max(g, dim=2)  # 计算每个特征维度的最大值
        global_feature = global_feature.unsqueeze(2).repeat(1, 1, 1)  # 重复以匹配输入的 N
        # 拼接
        c = torch.cat([seg_part1, global_feature], 1)

        # 拼接后的卷积层
        c = F.tanh(self.bn3_1(self.fc3_1(c)))
        c = F.tanh(self.bn3_2(self.fc3_2(c)))
        c = F.tanh(self.bn3_3(self.fc3_3(c)))
        c = F.tanh(self.bn3_4(self.fc3_4(c)))

        # 输出层
        prediction = self.prediction(c)

        prediction = prediction.squeeze(2)

        return prediction
