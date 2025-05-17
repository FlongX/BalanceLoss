import os
import torch
import random
import logging
import shutil
import numpy as np
from tqdm import tqdm
from utils import get_logger
import torch.multiprocessing
from net.unet_model import UNet
from config import create_parser
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from Dataset.pancreas import NihDataset
from Dataset.KiTS19 import KiTS19Dataset
from Dataset.BUSI import BUSIDataset
from Dataset.CVC import CVCDataset
from Dataset.KiTS19Tumor import KiTS19TumorDataset
from torchvision.utils import save_image
from EvaluationMetrics.EM import test_single_volume
from DataAugmentation.data_aug import RandomGenerator
torch.multiprocessing.set_sharing_strategy('file_system')


def inference(args, model, device):
    db_test = args.dataset(base_dir=args.root_path, split="test", outsize=args.img_size, transform=transforms.Compose(
        [RandomGenerator(output_size=args.img_size)]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    writer = SummaryWriter('./test_log/' + args.exp)
    recall_sum = 0
    dice_sum = 0
    precision_sum = 0
    iou_sum = 0
    hd_sum = 0

    model.eval()

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        recall, dice, precision, iou, hd95, predict = test_single_volume(image_batch, label_batch, model, device)

        image_batch = image_batch[0, 0:1, :, :]
        image_batch = (image_batch - image_batch.min()) / (image_batch.max() - image_batch.min())
        writer.add_image('test/Image', image_batch, i_batch)

        pred = torch.from_numpy(predict)
        pred = pred.unsqueeze(0)
        writer.add_image('test/Prediction', pred * 50, i_batch)

        labs = label_batch * 50
        writer.add_image('test/GroundTruth', labs, i_batch)
        save_image(image_batch, r'./test_log/' + 'image/' + str(i_batch) + '.png')
        save_image(pred.float(),
                   r'./test_log/' + 'pred/' + str(i_batch) + '.png')
        save_image(labs.float(),
                   r'./test_log/' + 'gt/' + str(i_batch) + '.png')
        logging.info('idx %d  recall %f dice %f precision %f iou %f' % (i_batch, recall, dice, precision, iou))

        recall_sum += recall
        dice_sum += dice
        precision_sum += precision
        iou_sum += iou
        hd_sum += hd95

    recall = recall_sum / len(db_test)
    dice = dice_sum / len(db_test)
    precision = precision_sum / len(db_test)
    iou = iou_sum / len(db_test)
    hd = hd_sum / len(db_test)

    logging.info('mean_recall %f mean_dice %f mean_precision %f mean_iou %f hd %f' % (recall, dice, precision, iou, hd))


if __name__ == "__main__":

    parer = create_parser()
    args = parer.parse_args()

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
    dataset_config = {

        'Pancreas': {'dataset': NihDataset, 'root_path': r'./Data/Pancreas/', 'num_classes': 2, 'img_size': [224, 224]},
        'KiTS19': {'dataset': KiTS19Dataset, 'root_path': r'./Data/kits19/', 'num_classes': 2, 'img_size': [224, 224]},
        'KiTS19Tumor': {'dataset': KiTS19TumorDataset, 'root_path': r'./Data/kits19/', 'num_classes': 2, 'img_size': [224, 224]},
        'BUSI': {'dataset': BUSIDataset, 'root_path': r'./Data/BUSI/', 'num_classes': 2, 'img_size': [224, 224]},
        'CVC-ClinicDB': {'dataset': CVCDataset, 'root_path': r'./Data/CVC-ClinicDB/', 'num_classes': 2, 'img_size': [224, 224]}
    }
    args.num_classes = dataset_config[args.Dataset_name]['num_classes']
    args.root_path = dataset_config[args.Dataset_name]['root_path']
    args.dataset = dataset_config[args.Dataset_name]['dataset']
    args.img_size = dataset_config[args.Dataset_name]['img_size']
    args.exp = 'U_' + args.Dataset_name + str(args.img_size)

    device = 1

    args.exp = args.model + '_' + args.Dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, args.model)
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)
    net = UNet(n_channels=3, n_classes=args.num_classes).cuda(device=device)
    net.load_state_dict(torch.load(snapshot_path + '/best_model.pth', map_location=torch.device(device=device)))

    if not os.path.exists(r'./test_log/'):
        os.makedirs(r'./test_log/', exist_ok=True)
    else:
        shutil.rmtree(r'./test_log/')
        os.makedirs(r'./test_log/', exist_ok=True)
        os.makedirs(r'./test_log/' + 'image/', exist_ok=True)
        os.makedirs(r'./test_log/' + 'pred/', exist_ok=True)
        os.makedirs(r'./test_log/' + 'gt/', exist_ok=True)
        os.makedirs(r'./test_log/' + args.exp, exist_ok=True)

    log_folder = './test_log/' + args.exp
    os.makedirs(log_folder, exist_ok=True)

    logger = get_logger(log_folder + '/' + 'log.txt')
    logger.info(str(args))

    inference(args, net, device)
