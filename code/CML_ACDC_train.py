import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import net_factory
from utils import val_2d
from utils.CML_utils import generate_mask_2D, features_discrepancy_loss, supervison_loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='CML', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--train_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per iteration')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
# -- setting of CML
parser.add_argument('--l_weight', type=float, default=1.0, help='weight of labeled pixels')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--dis_weight', type=float, default=0.1, help='weight of feature-level discrepancy loss')
parser.add_argument('--mask_ratio', type=float, default=1/2, help='ratio of mask/image')
args = parser.parse_args()

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    train_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    labeled_sub_bs, _ = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
    model = net_factory(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
							
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    #db_val = BaseDataSets(base_dir=args.root_path, split="test")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    iter_num = 0
    max_epoch = train_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                img_mask, _ = generate_mask_2D(img_a, args.mask_ratio)

            # CutMix input and label
            mix_input = img_a * img_mask + img_b * (1 - img_mask)
            mix_lab = lab_a * img_mask + lab_b * (1 - img_mask)

            # compute two model outputs
            outputs1, outputs2, feature1, feature2 = model(mix_input)

            # for one sub model
            loss_sup = supervison_loss(outputs1, mix_lab, class_num=args.num_classes)
            loss_dis = features_discrepancy_loss(feature1, feature2)

            # for another sub model
            sub_loss_sup = supervison_loss(outputs2, mix_lab, class_num=args.num_classes)
            sub_loss_dis = features_discrepancy_loss(feature2, feature1)

            # compute the overall loss
            loss_l = args.l_weight * (loss_sup + sub_loss_sup)
            loss_dis = args.dis_weight * (loss_dis + sub_loss_dis)
            loss = loss_l + loss_dis

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('pre/loss_l', loss_sup, iter_num)
            writer.add_scalar('pre/loss_dis', loss_dis, iter_num)
            writer.add_scalar('pre/loss', loss, iter_num)
            logging.info(
                'iteration %d : loss: %03f, loss_sup: %03f, loss_dis: %03f' % (iter_num, loss, loss_l, loss_dis))

            if iter_num % 200 == 1:
                image = mix_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                mix_outputs = (outputs1 + outputs2) / 2
                outputs = torch.argmax(torch.softmax(mix_outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = mix_lab[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)
                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= train_iterations:
                break
        if iter_num >= train_iterations:
            iterator.close()
            break
    writer.close()

def train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    train_iterations = args.train_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model = net_factory(in_chns=1, class_num=num_classes)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
							
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    #db_val = BaseDataSets(base_dir=args.root_path, split="test")
	
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    iter_num = 0
    max_epoch = train_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            ulab_a, ulab_b = label_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], label_batch[args.labeled_bs + unlabeled_sub_bs:]

            # for labeled data
            with torch.no_grad():
                img_mask, _ = generate_mask_2D(img_a, args.mask_ratio)

            """CutMix labeled input"""
            mixl_img = img_a * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + lab_b * (1 - img_mask)
            outputs1_l, outputs2_l, features1_l, features2_l = model(mixl_img)

            # for one sub model
            loss_sup_l = supervison_loss(outputs1_l, mixl_lab, class_num=args.num_classes)
            loss_dis_l = features_discrepancy_loss(features1_l, features2_l)

            # for another sub model
            sub_loss_sup_l = supervison_loss(outputs2_l, mixl_lab, class_num=args.num_classes)
            sub_loss_dis_l = features_discrepancy_loss(features2_l, features1_l)

            # The supervision objective for labeled data
            loss_l = args.l_weight * (loss_sup_l + sub_loss_sup_l)
            loss_dis_l = loss_dis_l + sub_loss_dis_l

            # for unlabeled data, heterogeneous mutual learning
            with torch.no_grad():
                unoutputs1, unoutputs2, _, _ = model(volume_batch[args.labeled_bs:])
                # get pseudo label
                plab = get_ACDC_masks(unoutputs1, nms=1)
                plab_sub = get_ACDC_masks(unoutputs2, nms=1)
                unimg_mask, _ = generate_mask_2D(unimg_a, args.mask_ratio)

            mixu_img = unimg_a * unimg_mask + unimg_b * (1 - unimg_mask)
            mixu_lab = ulab_a * unimg_mask + ulab_b * (1 - unimg_mask)
            # Supervise the cutmix portion with the sub model's pseudo label, and the rest is self-supervised.
            mixu_plab = plab_sub[:unlabeled_sub_bs] * unimg_mask + plab[unlabeled_sub_bs:] * (1 - unimg_mask)
            mixu_plab_sub = plab[:unlabeled_sub_bs] * unimg_mask + plab_sub[unlabeled_sub_bs:] * (1 - unimg_mask)

            # two model outputs
            outputs1_u, outputs2_u, features1_u, features2_u = model(mixu_img)
            # for one sub model
            loss_sup_u = supervison_loss(outputs1_u, mixu_plab, class_num=args.num_classes)
            loss_dis_u = features_discrepancy_loss(features1_u, features2_u)

            # for another sub model
            sub_loss_sup_u = supervison_loss(outputs2_u, mixu_plab_sub, class_num=args.num_classes)
            sub_loss_dis_u = features_discrepancy_loss(features2_u, features1_u)

            # The heterogeneous consistency objective for unlabeled data
            loss_u = args.u_weight * (loss_sup_u + sub_loss_sup_u)
            loss_dis_u = loss_dis_u + sub_loss_dis_u
            # The feature_level discrepancy loss for all data
            loss_dis = args.dis_weight * (loss_dis_l + loss_dis_u)

            # The total loss
            loss = loss_l + loss_u + loss_dis

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_dis', loss_u, iter_num)
            writer.add_scalar('Self/loss', loss, iter_num)
            logging.info('iteration %d : loss: %03f, loss_l: %03f, loss_u: %03f, loss_dis: %03f' % (iter_num, loss, loss_l, loss_u, loss_dis))

            if iter_num % 200 == 1:
                image_l = mixl_img[1, 0:1, :, :]
                writer.add_image('train/in_Image', image_l, iter_num)
                outputs_l = (outputs1_l + outputs2_l) / 2
                outputs = torch.argmax(torch.softmax(outputs_l, dim=1), dim=1, keepdim=True)
                writer.add_image('train/in_Prediction', outputs[1, ...] * 50, iter_num)
                labs = mixl_lab[1, ...].unsqueeze(0) * 50
                writer.add_image('train/in_GroundTruth', labs, iter_num)

                image_u = mixu_img[1, 0:1, :, :]
                writer.add_image('train/in_Image', image_u, iter_num)
                outputs_u = (outputs1_u + outputs2_u) / 2
                outputs = torch.argmax(torch.softmax(outputs_u, dim=1), dim=1, keepdim=True)
                writer.add_image('train/in_Prediction', outputs[1, ...] * 50, iter_num)
                labs = mixu_lab[1, ...].unsqueeze(0) * 50
                writer.add_image('train/in_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= train_iterations:
                break
        if iter_num >= train_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    pre_snapshot_path = "./model/CML/ACDC_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    train_snapshot_path = "./model/CML/ACDC_{}_{}_labeled/train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, train_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/CML_ACDC_train.py', train_snapshot_path)
    logging.basicConfig(filename=train_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # -- Pre_train --
    print("Starting CML pre-training.")
    pre_train(args, pre_snapshot_path)
    # -- train --
    print("Starting CML training.")
    train(args, pre_snapshot_path, train_snapshot_path)

    


