import os
import torch
import shutil
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from EvaluationMetrics.EM import test_single_volume
from torch.nn.modules.loss import CrossEntropyLoss
from DataAugmentation.data_aug import RandomGenerator
from utils import get_logger, gard, list2csv, show_reward_curve, pc_image, EarlyStopping
from loss.loss import FocalLoss, TopKLoss, TverskyLoss, AsymmetricUnifiedFocalLoss, DiceLoss, \
    InterCBLoss,TTopKLoss, IntraCBLoss, ComboLoss, CrossEntropyWithL1, show_pc


def trainer(args, model, snapshot_path, device):
    logger = get_logger(snapshot_path + "/log.txt")
    base_lr = args.base_lr
    lr_ = base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_epoch = args.max_epochs
    iter_num = 0
    a = 0.5
    trans = False
    iterator = tqdm(range(max_epoch), ncols=70)
    gard_b = []
    gard_b_fp = []
    gard_f = []

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    focal_loss = FocalLoss()
    tversky_loss = TverskyLoss()
    topk_loss = TopKLoss()
    IntraCB_loss = IntraCBLoss()
    ttopk_loss = TTopKLoss()
    InterCB_loss = InterCBLoss()
    rce_loss = CrossEntropyWithL1()
    ufl_loss = AsymmetricUnifiedFocalLoss()
    combo_loss = ComboLoss()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ES = EarlyStopping(patience=20, patience_lr=10)

    writer = SummaryWriter(snapshot_path + '/log')
    if not os.path.exists(snapshot_path + '/p/'):
        os.makedirs(snapshot_path + '/p/')

    db_train = args.dataset(base_dir=args.root_path, split="train", outsize=args.img_size,
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=args.img_size, split='train')]))
    db_val = args.dataset(base_dir=args.root_path, split="val", outsize=args.img_size,
                          transform=transforms.Compose(
                              [RandomGenerator(output_size=args.img_size)]))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=8)

    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1

    logger.info(str(args) + '\nThe length of train set is: {}\n\n===========================\n'.format(len(db_train)))
    logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    for epoch_num in iterator:
        model.train()
        for sampled_batch in trainloader:
            image_batch, label_batch, f_n, b_n = sampled_batch['image'], sampled_batch['label'], \
                sampled_batch['f_n'], sampled_batch['b_n']

            if 0 in label_batch.sum(dim=(1, 2)):
                print('have 0--\n')
                continue
            image_batch, label_batch, f_n, b_n = image_batch.cuda(device=device), label_batch.cuda(
                device=device), f_n.cuda(device=device), b_n.cuda(device)
            seg_out = model(image_batch)
            # loss_dice = dice_loss(seg_out, label_batch)
            # loss_ce = ce_loss(seg_out, label_batch[:].long())
            # loss_focal = focal_loss(seg_out, label_batch)
            # loss_combo = combo_loss(seg_out, label_batch)
            # loss_tversky = tversky_loss(seg_out, label_batch)
            # loss_ufl = ufl_loss(seg_out, label_batch)
            # loss_rce = rce_loss(seg_out, f_n, label_batch)
            # loss_topk = topk_loss(seg_out, label_batch, args.img_size)
            # loss_ttopk = ttopk_loss(seg_out, label_batch, device=device)

            if not trans:
                loss_IntraCB, trans = IntraCB_loss(seg_out, f_n, label_batch, batch_size, trans, device=device)
                loss = loss_IntraCB
            else:
                loss_InterCB = InterCB_loss(seg_out, f_n, label_batch, args.img_size, device=device)
                loss_IntraCB, trans = IntraCB_loss(seg_out, f_n, label_batch, batch_size, trans, device=device)
                loss = a*loss_IntraCB +(1-a)*loss_InterCB

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pc_f, pc_b, pc_b_a = show_pc(seg_out, label_batch, f_n)
            g_b, g_b_fp, g_f = gard(seg_out, f_n, b_n, label_batch, iter_num, len(trainloader), epoch_num)
            if iter_num % 50 == 0:
                gard_f.append(g_f)
                gard_b.append(g_b)
                if iter_num < 150:
                    gard_b_fp.append(0)
                else:
                    gard_b_fp.append(g_b_fp)
                pc_image(pc_f, pc_b, pc_b_a, iter_num)

            iter_num = iter_num + 1

            pred = torch.softmax(seg_out, dim=1)
            if iter_num < 1000:
                save_image(pred[0, 1, :, :].float(), snapshot_path + '/p/' + str(iter_num) + '.png')

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logger.info('iteration %d : gard_f : %.4f gard_b : %.4f gard_b_fp : %.4f  loss: %.4f' % (
                iter_num, g_f, g_b, g_b_fp, loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image_g', image, iter_num)

                outputs = torch.argmax(torch.softmax(seg_out, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)

                labs_g = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth_g', labs_g, iter_num)
        if iter_num > 150:
            list2csv(gard_b, column_name='gard_b', index_name='batch', csv_save_path=snapshot_path + '/gard_b.csv')
            list2csv(gard_b_fp, column_name='gard_bfp', index_name='batch',
                     csv_save_path=snapshot_path + '/gard_b_fp.csv')
            list2csv(gard_f, column_name='gard_f', index_name='batch', csv_save_path=snapshot_path + '/gard_f.csv')

            show_reward_curve(csv_path=snapshot_path + '/gard_b.csv',
                              png_save_path=snapshot_path + '/gard_b.png', xlab='Batch', ylab='gard_b',
                              Smoothing=100000)
            show_reward_curve(csv_path=snapshot_path + '/gard_b_fp.csv',
                              png_save_path=snapshot_path + '/gard_b_fp.png', xlab='Batch', ylab='gard_b_fp',
                              Smoothing=100000)
            show_reward_curve(csv_path=snapshot_path + '/gard_f.csv',
                              png_save_path=snapshot_path + '/gard_f.png', xlab='Batch', ylab='gard_f',
                              Smoothing=100000)
        model.eval()
        dice_sum = 0
        for sampled_batch in valloader:
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            _, dice, _, _, _, _ = test_single_volume(image_batch, label_batch, model, device)
            dice_sum += dice

        val_dice = dice_sum / len(db_val)
        logger.info("val_dice {}".format(val_dice))

        decay, early_stop, save_model = ES.EStop(val_dice)

        if save_model:
            torch.save(model.state_dict(), os.path.join(snapshot_path,'epoch_' + str(epoch_num) + '.pth'))
            logger.info("save model to {}".format(os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')))
        if early_stop:
            logger.info('EarlyStopping')
            break
        lr_ = lr_ * decay
        if lr_ < 1e-6:
            lr_ = 1e-6

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        logger.info(lr_)

    iterator.close()
    writer.close()
    return "Training Finished!"
