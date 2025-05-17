import os
import torch
import shutil
import random
import numpy as np
from trainer import trainer
from net.unet_model import UNet
from config import create_parser
import torch.backends.cudnn as cudnn
from Dataset.pancreas import NihDataset
from Dataset.KiTS19 import KiTS19Dataset
from Dataset.BUSI import BUSIDataset
from Dataset.CVC import CVCDataset
from Dataset.KiTS19Tumor import KiTS19TumorDataset

'''参数'''
parer = create_parser()
args = parer.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.Dataset_name = 'CVC-ClinicDB'
    args.model = 'UNet'
    device = 0
    dataset_config = {
        'Pancreas': {'dataset': NihDataset, 'root_path': r'./Data/Pancreas/', 'num_classes': 2, 'img_size': [224, 224]},
        'KiTS19': {'dataset': KiTS19Dataset, 'root_path': r'./Data/kits19/', 'num_classes': 2, 'img_size': [224, 224]},
        'KiTS19Tumor': {'dataset': KiTS19TumorDataset, 'root_path': r'./Data/kits19/', 'num_classes': 2, 'img_size': [224, 224]},
        'BUSI': {'dataset': BUSIDataset, 'root_path': r'./Data/BUSI', 'num_classes': 2},
        'CVC-ClinicDB': {'dataset': CVCDataset, 'root_path': r'./Data/CVC-ClinicDB/', 'num_classes': 2,
                         'img_size': [224, 224]},
    }

    if args.batch_size != 24:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[args.Dataset_name]['num_classes']
    args.root_path = dataset_config[args.Dataset_name]['root_path']
    args.dataset = dataset_config[args.Dataset_name]['dataset']
    args.img_size = dataset_config[args.Dataset_name]['img_size']

    args.exp = args.model + '_' + args.Dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, args.model)
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    if not os.path.exists(r'./model/'):
        os.makedirs(r'./model/', exist_ok=True)
    else:
        shutil.rmtree(r'./model/')
        os.makedirs(r'./model/pc/', exist_ok=True)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    net = UNet(n_channels=3, n_classes=args.num_classes).cuda(device=device)

    trainer(args, net, snapshot_path, device)
