import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import LeNet5
from torch import nn
from torch import optim
import time

def main():
    # 定义批次大小
    batchsz = 32

    # 数据预处理，将图像调整为32x32大小并转换为Tensor
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # 加载训练集
    cifar_train = datasets.CIFAR10('cifar', True, transform=transform, download=True)
    cifar_train_loader = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    # 加载测试集
    cifar_test = datasets.CIFAR10('cifar', False, transform=transform, download=True)
    cifar_test_loader = DataLoader(cifar_test, batch_size=batchsz, shuffle=False)  # 测试集不需要打乱顺序

    # 查看数据形状
    x, label = next(iter(cifar_train_loader))
    print('x:', x.shape, 'label:', label.shape)

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5.LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    # 训练循环
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        start_time = time.time()
        train_loss = 0.0
        for batchidx, (x, label) in enumerate(cifar_train_loader):
            x, label = x.to(device), label.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(cifar_train_loader)
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s')

        # 测试循环
        if (epoch + 1) % 10 == 0:  # 每10个epoch进行一次测试
            model.eval()  # 设置模型为评估模式
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, label in cifar_test_loader:
                    x, label = x.to(device), label.to(device)

                    logits = model(x)
                    loss = criterion(logits, label)

                    test_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1) # _是一个占位符，我们不关心第一个数据
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

            test_loss /= len(cifar_test_loader)
            accuracy = 100 * correct / total
            print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
