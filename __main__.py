import os
import torch
import CustomDataset
import torch.nn as nn
import torch.optim as optim
from models import NeuralNetwork
from torch.utils.data import DataLoader
import config


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork()
    model.to(device)
    dataset = CustomDataset.CustomDataset(image_dir=config.DATA_SET['TRAIN_PATH'],
                                          label_file=config.DATA_SET['TRAIN_FILE'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    model.train()

    for i in range(config.TRAIN_SET['num_epoc']):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"{i} / {config.TRAIN_SET['num_epoc'] - 1}, loss = {running_loss / len(dataloader)}")

    print("模型训练完成，你希望存储为什么名字")
    name = input()
    output_dir = config.MAIN_SET['OUTPUT_PATH']
    if not os.path.exists(config.MAIN_SET['OUTPUT_PATH']):
        os.makedirs(output_dir)
    output_dir = f"{output_dir}{name}.pth"
    torch.save(model, output_dir)
    print("存储完毕")


def test_model():
    print("请输入你想读取的模型名称")
    name = input()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f"./saved_models/{name}.pth", weights_only=False, map_location=device)

    model.eval()
    model.to(device)

    dataset = CustomDataset.CustomDataset(image_dir=config.DATA_SET['TEST_PATH'],
                                          label_file=config.DATA_SET['TEST_FILE'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.view(-1, 5, 36)
            for output, label in zip(outputs, labels):
                predicted = CustomDataset.tensor_transform_label(output)
                label = CustomDataset.tensor_transform_label(label)
                if label == predicted:
                    print(f"预测为{predicted}，实际为{label}，结果正确")
                    correct += 1
                else:
                    print(f"预测为{predicted}，实际为{label}，结果错误")
                total += 1
    print("测试完成")
    print(f"预测成功数量为{correct}，总预测数为{total}预测准确率为{correct / total}")


def main():
    mode = eval(input())
    if mode == 1:
        print("开始训练模型")
        train_model()
        print("训练模型结束")
    elif mode == 2:
        print("开始测试模型")
        test_model()
        print("测试模型结束")
    else:
        print("无效指令")


if __name__ == '__main__':
    main()
