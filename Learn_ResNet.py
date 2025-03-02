import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义Basic Block，适用于ResNet18和ResNet34
# 用BasicBlock 类实现一个基本的残差块
class BasicBlock(nn.Module):
    """
    基本残差块类，用于ResNet18和ResNet34。
    该残差块包含两个3x3卷积层，以及一个可选的下采样捷径连接。
    """
    # 定义扩张因子，对于Basic Block，输出通道数与输入通道数相同，扩张因子为1
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """
        初始化基本残差块。

        参数:
        in_channels (int): 输入特征图的通道数。
        out_channels (int): 输出特征图的通道数。
        stride (int, 可选): 第一个卷积层的步长，默认为1。
        """
        # 调用父类nn.Module的构造函数
        super(BasicBlock, self).__init__()
        # 第一个卷积层，使用3x3卷积核，步长由参数stride指定，填充为1以保持特征图尺寸不变（除了步长改变的情况），不使用偏置项
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 第一个批量归一化层，用于加速网络训练和提高模型的稳定性
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层，使用3x3卷积核，步长为1，填充为1，不使用偏置项
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 第二个批量归一化层
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接准备，如果输入输出通道数不同或者步长不为1，需要进行下采样
        self.shortcut = nn.Sequential()
        # 当步长不为1或者输入通道数不等于输出通道数乘以扩张因子时，需要进行下采样
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                # 使用1x1卷积核进行下采样，调整通道数和特征图尺寸
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                # 对下采样后的特征图进行批量归一化
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入特征图。

        返回:
        torch.Tensor: 输出特征图。
        """
        # 输入x经过第一个卷积层和批量归一化层，然后通过ReLU激活函数
        out = F.relu(self.bn1(self.conv1(x)))
        # 经过第二个卷积层和批量归一化层
        out = self.bn2(self.conv2(out))
        # 加上残差连接，将输入x经过shortcut处理后的结果与out相加
        out += self.shortcut(x) # 即 f(x) + x
        # 再次通过ReLU激活函数
        out = F.relu(out)
        return out

# 定义ResNet-18网络
class ResNet18(nn.Module):
    """
    ResNet-18模型类。
    该模型由一个初始卷积层、四个残差块阶段和一个全连接层组成。
    """
    def __init__(self, num_classes: int = 10) -> None:
        """
        初始化ResNet-18模型。

        参数:
        num_classes (int, 可选): 分类的类别数，默认为10。
        """
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
        self.layer1 = self._make_layer(64, 2, stride=1)
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

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        创建一个残差块阶段。

        参数:
        out_channels (int): 该阶段输出特征图的通道数。
        num_blocks (int): 该阶段包含的残差块数量。
        stride (int): 该阶段第一个残差块的步长。

        返回:
        nn.Sequential: 包含多个残差块的Sequential模块。
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法。

        参数:
        x (torch.Tensor): 输入图像张量。

        返回:
        torch.Tensor: 输出分类结果。
        """
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

    def _initialize_weights(self) -> None:
        """
        初始化模型的权重。
        卷积层使用Kaiming初始化，批量归一化层使用常量初始化。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4,
                         shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False,
                           download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4,
                        shuffle=False, num_workers=2)

# 定义类别
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 创建ResNet18模型实例
model = ResNet18(num_classes=10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
for epoch in range(2):  # 训练2个epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播 + 反向传播 + 优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个小批量打印一次
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
