import json
import torch
import torch.nn as nn
import random
import os
import joblib
import torchvision.transforms as transforms
from tqdm import tqdm
from models.helpers import load_checkpoint, test
from models.dataset_convert import custData
# 这个导入不能删，因为Proxy_dataset_convert和dataset_convert中的CustomDataset不一样，删掉会报错
# from models.Proxy_dataset_convert import CustomDataset


def evaluation(testloader, model, criterion, device=None, evaluation=True, is_proxy=False):
    if is_proxy:
        accuracy_bri, accuracy_con, _, precision_bri, recall_bri, precision_con, recall_con = \
                                         test(testloader, model, criterion, device=device, evaluation=True, is_proxy=is_proxy)
        print("Evaluation accuracy bri: {}%".format(accuracy_bri * 100))
        print("Evaluation accuracy con: {}%".format(accuracy_con * 100))
        # print("Evaluation precision bri: {}".format(precision_bri * 100))
        # print("Evaluation precision con: {}".format(precision_con * 100))
        # print("Evaluation recall bri: {}".format(recall_bri * 100))
        # print("Evaluation recall con: {}".format(recall_con * 100))
        return accuracy_bri+accuracy_con
    else:
        accuracy, _, precision, recall, f1_score = test(testloader, model, criterion, device=device, evaluation=True, is_proxy=is_proxy)
        print("Evaluation accuracy: {}%".format(accuracy * 100))
        print("Evaluation precision: {}".format(precision * 100))
        print("Evaluation recall: {}".format(recall * 100))
        print("Evaluation f1-score: {}".format(f1_score * 100))
        return accuracy


def main():
    # 是否是筛选最优模型模式
    IS_FILTER_BEST_MODEL = False
    # 是否训练代理任务模型
    IS_PROXY = False
    print("Evaluation, Is proxy: {}".format(IS_PROXY))
    if IS_FILTER_BEST_MODEL:
        print('******You''re filtering best model now******')
    if IS_PROXY:
        print('Evaluating Proxy model now')

    if True:
        # 保存的模型的路径
        saved_model_path = 'G:\\PaperCode\\2025-1\\FSA\\reports\\models\\resnet18'
        # 定义参数文件路径
        project_params_file = 'G:\\PaperCode\\2025-1\\FSA\\parameters\\project.json'
        evaluation_params_file = 'G:\\PaperCode\\2025-1\\FSA\\parameters\\evaluation.json'
        # 设置划分好的数据集所在路径
        dataset_path = "G:\\PaperCode\\2025-1\\FSA\\dataset\\splited_data"
        data_save_path = "G:\\PaperCode\\2025-1\\FSA\\dataset\\splited_data"
    else:
        # 保存的模型的路径
        saved_model_path = r"/root/autodl-tmp/reports/models/resnet18"
        # 定义参数文件路径
        project_params_file = r"/root/autodl-tmp/parameters/project.json"
        evaluation_params_file = r"/root/autodl-tmp/parameters/evaluation.json"
        # 设置划分好的数据集所在路径
        dataset_path = r"/root/autodl-tmp/dataset/splited_data"
        data_save_path = r"/root/autodl-tmp/dataset/splited_data"
    # 数据集已划分好，这里无需这两个参数，随便给个字符串过去
    dataset_root = 'No_Need'
    csv_file = 'No_Need'

    # load parameters
    project_parameters = json.load(open(project_params_file))
    evaluation_parameters = json.load(open(evaluation_params_file))

    # set random seed and device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = project_parameters["random_seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    #  cuDNN 后端设为确定性模式，确保每次运行卷积、RNN 等 CUDA 运算时，算法选择、内核顺序、线程块划分都完全相同，
    #  从而消除随机性，便于复现实验结果
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 因为数据集是前期已经划分好的，所以这里两个路径直接给'none'就好了，让它不会报错即可
    if IS_PROXY:
        test_path_proxy = os.path.join(dataset_path, 'test_dataset_proxy.joblib')
        test_dataset = joblib.load(test_path_proxy)
    else:
        _, test_dataset, _ = custData(dataset_root, csv_file, transform=None, save_path=data_save_path)

    # num_workers设置为0避免多进程随机性
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.48738178,), (0.24540879,))  # 标准化图像
    ])
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=evaluation_parameters["batch_size"],
                                             shuffle=False, num_workers=0)

    if IS_FILTER_BEST_MODEL:
        file_names = os.listdir(saved_model_path)
        best_model = {'accuracy': 0.0, 'model': ''}
        for names in tqdm(file_names):
            # 清理前一个模型
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()

            model, training_parameters = load_checkpoint(saved_model_path+'\\'+names, evaluation=True)
            model = model.to(device)
            # 因为没有去评估evaluation的误差水平(没啥用的指标)，所以这个只是拿来凑数的。
            criterion = nn.CrossEntropyLoss()
            tmp = evaluation(testloader, model, criterion, device=device, evaluation=True, is_proxy=IS_PROXY)
            if tmp>best_model['accuracy']:
                best_model['accuracy'] = tmp
                best_model['model'] = names
        print('best model is {}, with accuracy {}'.format(best_model['model'], best_model['accuracy']))
    else:
        model, training_parameters = load_checkpoint(evaluation_parameters["model_path"], evaluation=True)
        model = model.to(device)
        # 因为没有去评估evaluation的误差水平(没啥用的指标)，所以这个只是拿来凑数的。
        criterion = nn.CrossEntropyLoss()
        _ = evaluation(testloader, model, criterion, device=device, evaluation=True, is_proxy=IS_PROXY)


if __name__ == "__main__":
    main()
