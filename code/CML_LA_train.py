import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn

from skimage.measure import label
from torch.utils.data import DataLoader
from utils import test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.CML_utils import generate_mask_3D, features_discrepancy_loss, supervison_loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='CML', help='model_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--train_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
# -- setting of CML
parser.add_argument('--l_weight', type=float, default=1.0, help='weight of labeled pixels')
parser.add_argument('--u_weight', type=float, default=1.0, help='weight of unlabeled pixels')
parser.add_argument('--dis_weight', type=float, default=0.2, help='weight of features discrepancy loss')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
args = parser.parse_args()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    return torch.Tensor(np.array(batch_list)).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

train_data_path = args.root_path
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
train_max_iterations = args.train_max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2

def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, _ = generate_mask_3D(img_a, args.mask_ratio)

            # CutMix input and label
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            # compute two model outputs
            outputs1, outputs2, feature1, feature2 = model(volume_batch)

            # for one sub model
            loss_sup = supervison_loss(outputs1, label_batch)
            loss_dis = features_discrepancy_loss(feature1, feature2)

            # for another sub model
            sub_loss_sup = supervison_loss(outputs2, label_batch)
            sub_loss_dis = features_discrepancy_loss(feature2, feature1)

            # compute the overall loss
            loss_l = args.l_weight * (loss_sup + sub_loss_sup)
            loss_dis = args.dis_weight * (loss_dis + sub_loss_dis)
            loss = loss_l + loss_dis

            iter_num += 1
            writer.add_scalar('pre/loss_l', loss_sup, iter_num)
            writer.add_scalar('pre/loss_dis', loss_dis, iter_num)
            writer.add_scalar('pre/loss', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(
                'iteration %d : loss: %03f, loss_sup: %03f, loss_dis: %03f' % (iter_num, loss, loss_l, loss_dis))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()
            if iter_num >= pre_max_iterations:
                break
        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def train(args, pre_snapshot_path, train_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net_opt(model, optimizer, pretrained_model)

    model.train()
    writer = SummaryWriter(train_snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = train_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + sub_bs], label_batch[args.labeled_bs + sub_bs:]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch[args.labeled_bs + sub_bs:]

            # for labeled data
            with torch.no_grad():
                img_mask, _ = generate_mask_3D(img_a, args.mask_ratio)

            # cutMix labeled input
            mixl_img = img_a * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + lab_b * (1 - img_mask)
            outputs1_l, outputs2_l, features1_l, features2_l = model(mixl_img)

            # for one sub model
            loss_sup_l = supervison_loss(outputs1_l, mixl_lab)
            loss_dis_l = features_discrepancy_loss(features1_l, features2_l)

            # for another sub model
            sub_loss_sup_l = supervison_loss(outputs2_l, mixl_lab)
            sub_loss_dis_l = features_discrepancy_loss(features2_l, features1_l)

            loss_l = args.l_weight * (loss_sup_l + sub_loss_sup_l)
            loss_dis_l = loss_dis_l + sub_loss_dis_l

            # for unlabeled data, heterogeneous mutual learning
            with torch.no_grad():
                unoutputs1, unoutputs2, _, _ = model(volume_batch[args.labeled_bs:])
                # get pseudo label
                plab = get_cut_mask(unoutputs1, nms=1)
                plab_sub = get_cut_mask(unoutputs2, nms=1)
                unimg_mask, _ = generate_mask_3D(unimg_a, args.mask_ratio)

            mixu_img = unimg_a * unimg_mask + unimg_b * (1 - unimg_mask)
            mixu_lab = ulab_a * unimg_mask + ulab_b * (1 - unimg_mask)
            # Supervise the cutmix portion with the sub model's pseudo label, and the rest is self-supervised.
            mixu_plab = plab_sub[:sub_bs] * unimg_mask + plab[sub_bs:] * (1 - unimg_mask)
            mixu_plab_sub = plab[:sub_bs] * unimg_mask + plab_sub[sub_bs:] * (1 - unimg_mask)

            # two model outputs
            outputs1_u, outputs2_u, features1_u, features2_u = model(mixu_img)
            # for one sub model
            loss_sup_u = supervison_loss(outputs1_u, mixu_plab)
            loss_dis_u = features_discrepancy_loss(features1_u, features2_u)

            # for another sub model
            sub_loss_sup_u = supervison_loss(outputs2_u, mixu_plab_sub)
            sub_loss_dis_u = features_discrepancy_loss(features2_u, features1_u)
 
            loss_u = args.u_weight * (loss_sup_u + sub_loss_sup_u)
            loss_dis_u = loss_dis_u + sub_loss_dis_u

            # compute the overall loss
            loss_dis = args.dis_weight * (loss_dis_l + loss_dis_u)
            loss = loss_l + loss_u + loss_dis

            iter_num += 1
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_dis', loss_dis, iter_num)
            writer.add_scalar('Self/loss', loss, iter_num)

            # optimize main model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, loss_dis: %03f' % (iter_num, loss, loss_l, loss_u, loss_dis))

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(train_snapshot_path,'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(train_snapshot_path, '{}_best_model.pth'.format(args.model))
                    #save_net_opt(model, optimizer, save_mode_path)
                    #save_net_opt(model, optimizer, save_best_path)
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num % 200 == 1:
                ins_width = 2
                outputs_l = (outputs1_l + outputs2_l) /2
                B,C,H,W,D = outputs_l.size()
                snapshot_img = torch.zeros(size = (D, 3, 3*H + 3 * ins_width, W + ins_width), dtype = torch.float32)

                snapshot_img[:,:, H:H+ ins_width,:] = 1
                snapshot_img[:,:, 2*H + ins_width:2*H + 2*ins_width,:] = 1
                snapshot_img[:,:, 3*H + 2*ins_width:3*H + 3*ins_width,:] = 1
                snapshot_img[:,:, :,W:W+ins_width] = 1

                outputs_l_soft = F.softmax(outputs_l, dim=1)
                seg_out = outputs_l_soft[0,1,...].permute(2,0,1) # y
                target =  mixl_lab[0,...].permute(2,0,1)
                train_img = mixl_img[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_labeled'% (epoch, iter_num), snapshot_img)

                outputs_u = (outputs1_u + outputs2_u) / 2
                outputs_u_soft = F.softmax(outputs_u, dim=1)
                seg_out = outputs_u_soft[0,1,...].permute(2,0,1) # y
                target =  mixu_lab[0,...].permute(2,0,1)
                train_img = mixu_img[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch, iter_num), snapshot_img)

            if iter_num >= train_max_iterations:
                break

        if iter_num >= train_max_iterations:
            iterator.close()
            break
    writer.close()

if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./model/CML/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    train_snapshot_path = "./model/CML/LA_{}_{}_labeled/train".format(args.exp, args.labelnum)
    print("Starting CML training.")
    for snapshot_path in [pre_snapshot_path, train_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/CML_LA_train.py', train_snapshot_path)
    # -- Pre-Training --
    logging.basicConfig(filename=pre_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)
    # -- training --
    logging.basicConfig(filename=train_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, pre_snapshot_path, train_snapshot_path)