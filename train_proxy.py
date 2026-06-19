import time
import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
import random
import json
import numpy as np
import joblib
from tqdm import tqdm
from models.helpers import save_checkpoint, load_checkpoint, train, test
from models.resnet import ResNetk
from torch import optim
from torch.utils.tensorboard import SummaryWriter
# 这个导入不能删，因为Proxy_dataset_convert和dataset_convert中的CustomDataset不一样，删掉会报错
from models.Proxy_dataset_convert import CustomDataset

# 0健康，1有病
# 用来训练代理任务

def training():
   # 是否训练代理任务模型
    IS_PROXY = True
    # 分类数量
    NUM_OF_CLASSES = 10
    print("Normal training, is proxy: {}. Num of classes: {}".format(IS_PROXY, NUM_OF_CLASSES))

    if True:
        project_params_file = 'G:\\PaperCode\\2025-1\\FSA\\parameters\\project.json'
        training_params_file = 'G:\\PaperCode\\2025-1\\FSA\\parameters\\training.json'
        # 设置图片数据集所在路径
        dataset_root = 'G:\\PaperCode\\2025-1\\FSA\\dataset'
        dataset_path = "G:\\PaperCode\\2025-1\\FSA\\dataset\\splited_data"
    else:
        project_params_file = r"/root/autodl-tmp/parameters/project.json"
        training_params_file = r"/root/autodl-tmp/parameters/training.json"
        # 设置图片数据集所在路径
        dataset_root = r"/root/autodl-tmp/dataset"
        # 数据集路径chuli
        dataset_path = r"/root/autodl-tmp/dataset/splited_data"

    # 数据集路径处理
    train_path_proxy = os.path.join(dataset_path, 'train_dataset_proxy.joblib')
    test_path_proxy = os.path.join(dataset_path, 'test_dataset_proxy.joblib')
    val_path_proxy = os.path.join(dataset_path, 'val_dataset_proxy.joblib')

    # load parameters
    project_parameters = json.load(open(project_params_file))
    training_parameters = json.load(open(training_params_file))

    # set random seed and device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    seed = project_parameters["random_seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # datasets and dataloaders
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45, expand=False),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.48738178,), (0.24540879,))  # 标准化图像
    ])

    # 载入数据集
    train_dataset = joblib.load(train_path_proxy)
    test_dataset = joblib.load(test_path_proxy)
    val_dataset = joblib.load(val_path_proxy)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = training_parameters["batch_size"],
                                              shuffle=True, num_workers= training_parameters["num_workers"])
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size = training_parameters["batch_size"],
                                              shuffle=False, num_workers= training_parameters["num_workers"])

    net = ResNetk(
        num_classes=NUM_OF_CLASSES,
        k=training_parameters["resnet_depth"],
        is_proxy=IS_PROXY
        ).to(device)

    epoch_start = 0
    optimizer = optim.Adam(net.parameters(), lr=training_parameters["lr"])

    criterion = nn.CrossEntropyLoss()

    # losses
    epoch_losses = []
    # 生成tensorboard用以记录数据
    writer = SummaryWriter(log_dir=project_parameters["losses"])

    s = time.time()
    for epoch in tqdm(range(epoch_start, training_parameters["n_epochs"])):
        batch_losses = train(trainloader, optimizer, net, criterion, training_parameters, epoch, device, writer,
                             is_proxy=IS_PROXY)
        accuracy_bri, accuracy_con, test_loss_avg, _, _, _, _ = test(valloader, net, criterion, device, epoch, is_proxy=IS_PROXY)

        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        print("Epoch n°{}, epoch loss: {}, bri accuracy: {}, con accuracy: {}, duration {}".format(
            epoch, epoch_loss, accuracy_bri, accuracy_con, time.time() - s))

        # 记录训练数据到tensorboard
        writer.add_scalar('TrainLoss/epoch', epoch_loss, epoch)
        # 记录测试 精度 数据到tensorboard
        writer.add_scalar('Test/ACC_bri', accuracy_bri, epoch)
        writer.add_scalar('Test/ACC_con', accuracy_con, epoch)
        # 记录测试 误差 数据到tensorboard
        writer.add_scalar('Test/loss_avg', test_loss_avg, epoch)

        # save model every save_every epoch
        if epoch > 20 and epoch % training_parameters["save_every"] == 0:
            save_checkpoint(net, optimizer, training_parameters, project_parameters["models"], epoch)

    writer.close()


if __name__ == '__main__':
    training()

