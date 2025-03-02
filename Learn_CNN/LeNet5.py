import torch
from torch import nn

class LeNet5(nn.Module):
    """
    for cifar10 dataset
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 6, 28, 28]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),  # [b, 6, 28, 28] => [b, 6, 14, 14]
            # [b, 6, 14, 14] => [b, 16, 10, 10]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # [b, 16, 10, 10] => [b, 16, 5, 5]
        )
        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 修正输入维度
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        # 去掉打印语句，避免每次初始化模型时都打印
        # tmp = torch.randn(2, 3, 32, 32)
        # out = self.conv_unit(tmp)
        # print('conv out:', out.shape)

    def forward(self, x):
        """
        :param
        x: [b, 3, 32, 32]
        :return:
        """
        batchsz = x.size(0)
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, -1)  # 使用-1自动计算维度
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)
        return logits

def main():
    net = LeNet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('conv out:', out.shape)
