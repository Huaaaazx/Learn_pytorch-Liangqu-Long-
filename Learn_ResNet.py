import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义Basic Block，适用于ResNet18和ResNet34
# 用BasicBlock 类实现一个基本的残差块
class BasicBlock(nn.Module):
    # 定义扩张因子，对于Basic Block，输出通道数与输入通道数相同，扩张因子为1
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        # 调用父类nn.Module的构造函数
        super(BasicBlock, self).__init__()
        # 第一个卷积层，使用3x3卷积核，步长由参数stride指定，填充为1以保持特征图尺寸不变（除了步长改变的情况），不使用偏置项
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # stride=stride表示自定义步长;padding=1 意味着在输入特征图的上下左右各填充 1 个像素的宽度;因为后面有归一化层所以不需要偏置
        # 第一个批量归一化层，用于加速网络训练和提高模型的稳定性
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层，使用3x3卷积核，步长为1，填充为1，不使用偏置项
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 第二个批量归一化层
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接准备，如果输入输出通道数不同或者步长不为1，需要进行下采样
        self.shortcut = nn.Sequential()
        # 这里只是初始化了一个空的 nn.Sequential 容器，它为后续根据不同情况构建残差连接做好准备
        # 当步长不为1或者输入通道数不等于输出通道数乘以扩张因子时，需要进行下采样
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                # 使用1x1卷积核进行下采样，调整通道数和特征图尺寸
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                # 对下采样后的特征图进行批量归一化
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        # 输入x经过第一个卷积层和批量归一化层，然后通过ReLU激活函数
        out = F.relu(self.bn1(self.conv1(x)))
        # 经过第二个卷积层和批量归一化层
        out = self.bn2(self.conv2(out))
        # 加上残差连接，将输入x经过shortcut处理后的结果与out相加
        out += self.shortcut(x) #（如果 self.shortcut 为空，就直接返回 x；如果不为空，则进行相应的调整）
        # 再次通过ReLU激活函数
        out = F.relu(out)
        return out

# 定义ResNet-18网络
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        # 调用父类nn.Module的构造函数
        super(ResNet18, self).__init__()
        # 初始的输入通道数
        self.in_channels = 64

        # 第一个卷积层，输入通道数为3（对应RGB图像），输出通道数为64，使用3x3卷积核，步长为1，填充为1，不使用偏置项
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 第一个批量归一化层
        self.bn1 = nn.BatchNorm2d(64)
        # 四个阶段的残差块
        # 第一个阶段，输出通道数为64，包含2个残差块，步长为1
        self.layer1 = self._make_layer(64, 2, stride=1) # _make_layer的定义在后面
        # 第二个阶段，输出通道数为128，包含2个残差块，步长为2
        self.layer2 = self._make_layer(128, 2, stride=2)
        # 第三个阶段，输出通道数为256，包含2个残差块，步长为2
        self.layer3 = self._make_layer(256, 2, stride=2)
        # 第四个阶段，输出通道数为512，包含2个残差块，步长为2
        self.layer4 = self._make_layer(512, 2, stride=2)
        # 全局平均池化层，将特征图的每个通道的所有元素求平均，得到一个标量
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，输入维度为512乘以扩张因子（即512），输出维度为类别数
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        # 定义每个阶段的步长列表，第一个残差块的步长由参数stride指定，其余残差块的步长为1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # 依次添加残差块到layers列表中
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            # 更新输入通道数为当前阶段的输出通道数乘以扩张因子
            self.in_channels = out_channels * BasicBlock.expansion
        # 将layers列表中的模块组合成一个Sequential模块
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入x经过第一个卷积层和批量归一化层，然后通过ReLU激活函数
        out = F.relu(self.bn1(self.conv1(x)))
        # 经过第一个阶段的残差块
        out = self.layer1(out)
        # 经过第二个阶段的残差块
        out = self.layer2(out)
        # 经过第三个阶段的残差块
        out = self.layer3(out)
        # 经过第四个阶段的残差块
        out = self.layer4(out)
        # 经过全局平均池化层
        out = self.avg_pool(out)
        # 将特征图展平为一维向量
        out = out.view(out.size(0), -1)
        # 经过全连接层得到最终的分类结果
        out = self.fc(out)
        return out

# 测试代码
if __name__ == '__main__':
    # 创建一个ResNet18模型实例，类别数为10
    model = ResNet18(num_classes=10)
    # 生成一个随机的输入张量，形状为(1, 3, 32, 32)，表示1个样本，3个通道，32x32的图像
    input_tensor = torch.randn(1, 3, 32, 32)
    # 将输入张量传入模型，得到输出
    output = model(input_tensor)
    # 打印输出的形状
    print(output.shape)
