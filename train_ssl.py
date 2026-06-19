import json
import torch
import random
import os
import joblib
import torchvision.transforms as transforms
from models.diff_tasks import SslOnly, train_model
from torch.utils.tensorboard import SummaryWriter
from models.helpers import load_checkpoint


def training():
    # 最佳代理模型
    BEST_PROXY_MODEL = '112_resnet18'
    # 是否冻结backbone
    IS_BACKBONE_FREEZED = False
    # 是否有增强分支
    IS_KMEANS_BRANCH = True
    # 是否注意力机制
    IS_ATTENTION_MECH = True
    # 是否训练代理任务模型
    IS_PROXY = False
    # 分类数量
    NUM_OF_CLASSES = 2
    print("Normal training. Is proxy: {}. Num of classes: {}".format(IS_PROXY, NUM_OF_CLASSES))
    if not IS_BACKBONE_FREEZED:
        print('**********Backbone not freezed training***********')
    if IS_KMEANS_BRANCH:
        print('Training with K-means enchancement branch')
    if IS_ATTENTION_MECH:
        print('Training with Channel attention mechanism')

    if True:
        # 定义参数文件路径
        project_params_file = 'G:\\PaperCode\\2025-1\\FSA\\parameters\\project.json'
        training_params_file = 'G:\\PaperCode\\2025-1\\FSA\\parameters\\training.json'
        # 设置划分好的数据集所在路径
        dataset_path = "G:\\PaperCode\\2025-1\\FSA\\dataset\\splited_data"
        # 设定目标模型路径
        model_path = 'G:\\PaperCode\\2025-1\\FSA\\reports\\models\\best_proxy\\{}.pth'.format(BEST_PROXY_MODEL)
    else:
        # 定义参数文件路径
        project_params_file = r"/root/autodl-tmp/parameters/project.json"
        training_params_file = r"/root/autodl-tmp/parameters/training.json"
        # 设置划分好的数据集所在路径
        dataset_path = r"/root/autodl-tmp/dataset/splited_data"
        # 设定目标模型路径
        model_path = r"/root/autodl-tmp/reports/models/best_proxy/{}.pth".format(BEST_PROXY_MODEL)
    # load parameters
    project_parameters = json.load(open(project_params_file))
    training_parameters = json.load(open(training_params_file))

    # set random seed
    seed = project_parameters["random_seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # get dataset and loader
    train_path_ssl = os.path.join(dataset_path, 'train_dataset_down.joblib')
    val_path_ssl = os.path.join(dataset_path, 'val_dataset_down.joblib')
    train_dataset_ssl = joblib.load(train_path_ssl)
    val_dataset_ssl = joblib.load(val_path_ssl)

    trainloader_ssl = torch.utils.data.DataLoader(train_dataset_ssl, batch_size=training_parameters["batch_size"],
                                                  shuffle=True, num_workers=training_parameters["num_workers"])
    valloader_ssl = torch.utils.data.DataLoader(val_dataset_ssl, batch_size = training_parameters["batch_size"],
                                             shuffle= False, num_workers= training_parameters["num_workers"])

    # 载入代理任务模型
    model, optimizer, training_params, epoch = load_checkpoint(model_path, evaluation=False)
    # 规整为下游任务模型
    model = SslOnly(model, NUM_OF_CLASSES, is_kmeans=IS_KMEANS_BRANCH, is_att_mech=IS_ATTENTION_MECH)

    # 训练模型
    # 一定注意freeze_backbone参数的设置
    train_model(model, trainloader_ssl, valloader_ssl, freeze_backbone=IS_BACKBONE_FREEZED,\
                training_parameters=training_parameters, project_parameters=project_parameters,
                is_kmeans=IS_KMEANS_BRANCH, is_att_mech=IS_ATTENTION_MECH)


if __name__ == '__main__':
    training()
