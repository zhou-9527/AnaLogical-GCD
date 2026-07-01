import sys
import inspect
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import clip
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.cluster.hierarchy import linkage, fcluster
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import argparse
import os
from model import ALNetCLIP, ALNet
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from loguru import logger
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits

from project_utils.general_utils import  str2bool

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm

from torch.nn import functional as F

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path, dino_pretrain_path2
from matplotlib import pyplot as plt
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
import random
from kmeans_pytorch import kmeans

# TODO: Debug
import warnings
import math
warnings.filterwarnings("ignore")

def init_experiment(args, runner_name=None, exp_id=None):

    args.cuda = torch.cuda.is_available()

    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:

        # Unique identifier for experiment
        now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
              datetime.now().strftime("%S.%f")[:-3] + ')'
        log_dir = os.path.join(root_dir, "All_Samples", args.dataset_name + "-Acc" + now)
        while os.path.exists(log_dir):
            now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
                  datetime.now().strftime("%S.%f")[:-3] + ')'
            log_dir = os.path.join(root_dir, 'log', now)

    else:

        log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')


    args.writer = SummaryWriter(log_dir=args.log_dir)

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})


    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir
    args.logger.info(f'Experiment saved to: {args.log_dir}')

    args.logger.info(runner_name)
    args.logger.info(args)

    return args


def test_agglo_all(mask_test, epoch, feats, targets, mask, args):
    time_s =time.time()
    mask = mask.astype(bool)
    linked = linkage(feats, method="ward")
    gt_dist = linked[:, 2][-args.num_labeled_classes - args.num_unlabeled_classes]
    preds = fcluster(linked, t=gt_dist, criterion='distance')

    preds_select = preds[~mask_test]
    targets_select = targets[~mask_test]
    mask_select = mask[~mask_test]
    test_all_acc_test, test_old_acc_test, test_new_acc_test = log_accs_from_preds(y_true=targets_select, y_pred=preds_select,
                                                                                  mask=mask_select,
                                                                                  T=epoch, eval_funcs=args.test_eval_funcs,
                                                                                  save_name="Unlabeled Test")
    time_e = time.time()
    time_use = time_e - time_s
    print("Test Num: %d" % len(preds_select))
    print("Clustering time use: %.2f mins"%(time_use/60))
    return test_all_acc_test, test_old_acc_test, test_new_acc_test

def test_agglo_semisup(mask_test, epoch, feats, targets, mask, args):
    time_s = time.time()
    all_feats = feats
    l_feats = all_feats[mask_test]  # Get labelled set
    u_feats = all_feats[~mask_test]  # Get unlabelled set
    l_targets = targets[mask_test]  # Get labelled targets
    u_targets = targets[~mask_test]  # Get unlabelled targets
    n_samples = len(targets)

    if args.unbalanced:
        cluster_size = None
    else:
        cluster_size = math.ceil(n_samples / (args.num_labeled_classes + args.num_unlabeled_classes))
    kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None, cluster_size=cluster_size)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))
    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeanssem.labels_
    preds_select = all_preds.cpu().numpy()[~mask_test]
    targets_select = targets[~mask_test]
    mask_select = mask[~mask_test]

    test_all_acc_test, test_old_acc_test, test_new_acc_test = log_accs_from_preds(y_true=targets_select, y_pred=preds_select , mask=mask_select,
                                                                                    eval_funcs=args.test_eval_funcs,
                                                                                    save_name='SS-K-Means Train ACC Unlabelled')

    time_e = time.time()
    time_use = time_e - time_s
    print("Test Num: %d" % len(preds_select))
    print("Clustering time use: %.2f mins" % (time_use/60))
    return test_all_acc_test, test_old_acc_test, test_new_acc_test


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target, similarity,smoothing = 0.5):
        target_smooth = F.one_hot(target,input.size(1)).float()*(1-smoothing) +smoothing*similarity#F.one_hot(similarity,input.size(1)).float()#s1/input.size(0)#coef# / self.num_classes
        return torch.nn.CrossEntropyLoss()(input, target_smooth)




class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None,is_code=False):#, smoothing=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if is_code:
            dist = torch.cdist(anchor_feature, contrast_feature)
            dist=-dist/(dist.sum(dim=1)+1e-10)
        else:
            dist = -torch.cdist(anchor_feature, contrast_feature)

        anchor_dot_contrast = torch.div(dist, self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, confusion_factor, args,is_code=False):

    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    if is_code:
        dist=torch.cdist(features, features,p=2)
        similarity_matrix =-dist/(dist.sum(dim=1)+1e-10)
    else:
        similarity_matrix=-torch.cdist(features, features)


    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)


    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    pos_confs= confusion_factor[labels.bool()].view(confusion_factor.shape[0], -1)


    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    neg_confs= confusion_factor[~labels.bool()].view(confusion_factor.shape[0], -1)


    logits = torch.cat([positives, negatives], dim=1)
    log_confs = torch.cat([pos_confs, neg_confs], dim=1)

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels, log_confs


def test(vis_encoder, text_encoder, AL_model, test_loader, args, epoch):
    with torch.no_grad():
        vis_encoder.eval()
        AL_model.eval()
        text_encoder.eval()
        all_feats_val_norm = []
        targets = np.array([])
        mask = np.array([])
        mask_test = []
        all_uq_idxs_test = []
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images, labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0].bool()
            mask_test.append(mask_lab)
            images = images.type(clip_dtype).to(device)
            image_features = vis_encoder(images)

            text_features_labelled = text_encoder(text_tonkenized).to(device)
            text_feature_base = []
            for i in range(len(label_list)):
                text_feature_base.append(text_features_labelled[label_list[i]].unsqueeze(0))
            text_feature_base = torch.cat(text_feature_base, dim=0)
            text_features_AL = AL_model(image_features, vis_features_labelled, text_feature_base, args)

            vis_text_features = args.vis_rate * image_features + (1 - args.vis_rate) * text_features_AL
            features = vis_text_features
            all_feats_val_norm.append(torch.nn.functional.normalize(features, dim=-1).detach().cpu().numpy())
            targets = np.append(targets, labels.cpu().numpy())
            mask = np.append(mask, np.array(
                [True if x.item() in range(len(args.train_classes)) else False for x in labels]))
            all_uq_idxs_test.append(uq_idxs.cpu())
        mask_test = np.array(torch.cat(mask_test))
        all_feats_val_norm = np.concatenate(all_feats_val_norm)

    img_all_acc_test_norm, img_old_acc_test_norm, img_new_acc_test_norm = test_agglo_all(mask_test, epoch, all_feats_val_norm, targets, mask, args)
    args.logger.info('Know-K       Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f} | Eval_funcs: {}'.format(
        img_all_acc_test_norm, img_old_acc_test_norm, img_new_acc_test_norm, args.test_eval_funcs))
    img_all_acc_test, img_old_acc_test, img_new_acc_test = test_agglo_semisup(mask_test, epoch, all_feats_val_norm, targets, mask, args)
    args.logger.info('Know-K  Clu    Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f} | Eval_funcs: {}'.format(
            img_all_acc_test, img_old_acc_test, img_new_acc_test, args.test_eval_funcs))
    return img_all_acc_test_norm, img_old_acc_test_norm, img_new_acc_test_norm, img_all_acc_test, img_old_acc_test, img_new_acc_test


def train(projection_head, vis_encoder, text_encoder, AL_model, train_loader, test_loader, merge_train_loader, args):

    optimizer = SGD(list(projection_head.parameters()) + list(vis_encoder.parameters()) + list(text_encoder.parameters()) + list(AL_model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)


    sup_con_crit = SupConLoss()
    strategy=args.strategy
    cluster_momentum=args.cluster_momentum

    Total_loss=[]
    Contrastive_loss=[]

    best_semi_sup_acc = 0
    best_all_acc_norm = 0
    best_old_acc_norm = 0
    best_new_acc_norm = 0
    best_epoch_norm = 0

    best_all_acc = 0
    best_old_acc = 0
    best_new_acc = 0
    best_epoch = 0


    unsupervised_smoothing = args.unsupervised_smoothing
    train_report_interval = args.train_report_interval
    prototype_extraction_interval=args.prototype_extraction_interval
    Distance=args.distance


    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        train_acc_record = AverageMeter()
        loss_cons_record = AverageMeter()

        with torch.no_grad():
            if epoch%prototype_extraction_interval==0:
                uq_index, all_preds, cluster_protos_list, preds_ind_list, metrics, semi_sup_acc= \
                    extract_labeled_protos(vis_encoder, text_encoder, AL_model,  merge_train_loader, args=args)
                for i in range(len(preds_ind_list)):
                    preds_ind_list[i] = preds_ind_list[i].to(device).long()
                    cluster_protos_list[i] = cluster_protos_list[i].to(device)
                    if Distance == 'cosine':
                        cluster_protos_list[i]=cluster_protos_list[i]/torch.norm(cluster_protos_list[i],dim=1).unsqueeze(1)

                cluster_distances_list=[]
                cluster_radius_list=[]
                for i in range(len(preds_ind_list)):
                    if Distance=='euclidean':
                        cluster_distances = torch.cdist(cluster_protos_list[i], cluster_protos_list[i])
                    else:
                        cluster_distances = torch.matmul(cluster_protos_list[i], cluster_protos_list[i].T)

                    cluster_distances_list.append(cluster_distances.clone())
                    cluster_radius = \
                    (cluster_distances + torch.eye(cluster_distances.shape[0]).to(device) * cluster_distances.max()).min(dim=1)[0] / 2
                    cluster_radius_list.append(cluster_radius.clone())

        if epoch % train_report_interval == 0 or (epoch > 150 and semi_sup_acc > best_semi_sup_acc) or epoch > 195:
            with torch.no_grad():
                vis_encoder.eval()
                AL_model.eval()
                text_encoder.eval()
                img_all_acc_test_norm, img_old_acc_test_norm, img_new_acc_test_norm,img_all_acc_test, img_old_acc_test, img_new_acc_test = test(
                    vis_encoder, text_encoder, AL_model, test_loader, args, epoch)

            if img_all_acc_test_norm > best_all_acc_norm:
                best_all_acc_norm = img_all_acc_test_norm
                best_old_acc_norm = img_old_acc_test_norm
                best_new_acc_norm = img_new_acc_test_norm
                best_epoch_norm = epoch
                if semi_sup_acc > best_semi_sup_acc:
                    best_semi_sup_acc = semi_sup_acc

                torch.save({
                    'projection_head': projection_head.state_dict(),
                    'vis_encoder': vis_encoder.state_dict(),
                    'AL_model': AL_model.state_dict(),
                    'text_encoder': text_encoder.state_dict()}
                    , args.model_path[:-3] + f'_best.pt')
                print("Best model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

            if img_all_acc_test > best_all_acc:
                best_all_acc = img_all_acc_test
                best_old_acc = img_old_acc_test
                best_new_acc = img_new_acc_test
                best_epoch = epoch

        args.logger.info(
            'Best Report          Epoch:{:.0f} Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f} | Eval_funcs: {} '.format(
                best_epoch_norm, best_all_acc_norm, best_old_acc_norm, best_new_acc_norm, args.test_eval_funcs))
        args.logger.info(
            'Best Semisup Report  Epoch:{:.0f} Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f} | Eval_funcs: {}'.format(
                best_epoch, best_all_acc, best_old_acc, best_new_acc, args.test_eval_funcs))


        projection_head.train()
        vis_encoder.train()
        AL_model.train()
        text_encoder.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            text_feature_base = []
            text_features_labelled = text_encoder(text_tonkenized).to(device)
            for i in range(len(label_list)):
                text_feature_base.append(text_features_labelled[label_list[i]].unsqueeze(0))
            text_feature_base = torch.cat(text_feature_base, dim=0)

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(device)
            images = images.type(clip_dtype).to(device)

            image_features = vis_encoder(images)
            text_features_AL = AL_model(image_features, vis_features_labelled, text_feature_base, args)
            vis_text_features = args.vis_rate * image_features + (1 - args.vis_rate) * text_features_AL

            # Extract features with base model
            features = vis_text_features
            features = features.type(torch.float32).to(device)

            # Pass features through projection head
            features_1, features_2 = features.detach().chunk(2)
            all_features = torch.cat([features_1, features_2], dim=0)
            features = projection_head(features)
            # L2-normalize features
            features = torch.nn.functional.normalize(features, dim=-1)
            with torch.no_grad():

                confusion_factor=0
                if Distance == 'euclidean':
                    pair_dist = torch.cdist(all_features, all_features)
                else:
                    normalized_feats = all_features/torch.norm(all_features.unsqueeze(1))
                    pair_dist = torch.matmul(normalized_feats, normalized_feats.T)

                n_labeled = args.num_labeled_classes
                n_unlabeled = args.num_unlabeled_classes

                for i in range(len(preds_ind_list)):
                    cluster_labels=(preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]).clone()
                    cluster_indexer = F.one_hot(cluster_labels.long(), n_labeled +n_unlabeled ).float().T
                    n_labeled = max(int(n_labeled / 2), 1)
                    n_unlabeled = max(int(n_unlabeled / 2), 1)

                    cluster_indexer = torch.cat([cluster_indexer,cluster_indexer],dim=1)
                    n_samples = torch.sum(cluster_indexer, dim=1).unsqueeze(1)
                    n_samples[n_samples == 0] = 1

                    if Distance=='euclidean':
                        distance = torch.cdist(all_features, cluster_protos_list[i].float())
                    else:
                        normalized_feats = all_features / torch.norm(all_features.unsqueeze(1))
                        distance = torch.matmul(normalized_feats, cluster_protos_list[i].float().T)

                    cluster_radius_list[i]= (cluster_indexer*distance.T).sum(dim=1)/n_samples.squeeze()\
                                            *(1-cluster_momentum)+cluster_radius_list[i]* cluster_momentum

                    cluster_labels = torch.cat([cluster_labels,cluster_labels])
                    if Distance=='euclidean':
                        if strategy=='zero_one':
                            confusion_factor+=(pair_dist>2*cluster_radius_list[i][cluster_labels]).float()/2 ** i
                        elif strategy=='pair_dist':
                            confusion_factor += pair_dist / 2 ** i
                        elif strategy=='pair_cluster':
                            confusion_factor += distance[:,cluster_labels] / 2 ** i
                        else:
                            pass

                    else:
                        if strategy=='zero_one':
                            confusion_factor += (pair_dist< distance[:,cluster_labels]/2).float()/ 2 ** i
                        elif strategy == 'pair_dist':
                            confusion_factor += -pair_dist / 2 ** i
                        elif strategy=='pair_cluster':
                            confusion_factor += -distance[:,cluster_labels] / 2 ** i

            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = features
            confusion_factor = (confusion_factor - confusion_factor.min()) / (
                        confusion_factor.max() - confusion_factor.min() + 0.0000001)
            confusion_factor = confusion_factor / confusion_factor.sum(dim=1)
            torch.cuda.empty_cache()

            torch.cuda.empty_cache()
            contrastive_logits, contrastive_labels, similarity= info_nce_logits(features=con_feats, confusion_factor=confusion_factor, args=args)
            contrastive_loss = LabelSmoothingLoss()(contrastive_logits, contrastive_labels, similarity, unsupervised_smoothing)

            f1n, f2n = features.chunk(2)
            semisup_con_feats = torch.cat([f1n.unsqueeze(1), f2n.unsqueeze(1)], dim=1)
            # Supervised contrastive loss
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            dimension=semisup_con_feats.shape[-1]
            for i in range(len(preds_ind_list)):
                sup_con_loss += sup_con_crit(semisup_con_feats[:, :, :int(dimension/2**(i+1))],
                                             labels=preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]) / 2**(i+1)

            # Total loss
            loss_in = (1 - args.sup_con_weight) * (contrastive_loss) + \
                      args.sup_con_weight * (sup_con_loss)/2

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_cons_record.update(loss_in.item(), class_labels.size(0))
            loss = loss_in
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        args.logger.info('Epoch: {} Avg Loss: {:.3f} | Constrastive: {:.3f} '.format(epoch, loss_record.avg, loss_cons_record.avg))

        Total_loss.append(loss_record.avg)
        Contrastive_loss.append(loss_cons_record.avg)

        torch.save({
            'projection_head': projection_head.state_dict(),
            'vis_encoder': vis_encoder.state_dict(),
            'AL_model': AL_model.state_dict(),
            'text_encoder': text_encoder.state_dict()}
            , args.model_path[:-3] + f'_final.pt')
        print("Final model saved to {}.".format(args.model_path[:-3] + f'_final.pt'))


    args.logger.info('############# Final Reports #############')
    img_all_acc_test_norm, img_old_acc_test_norm, img_new_acc_test_norm, img_all_acc_test, img_old_acc_test, img_new_acc_test = test(
        vis_encoder, text_encoder, AL_model, test_loader, args, epoch)

    args.logger.info('Final Report         Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
        img_all_acc_test_norm, img_old_acc_test_norm, img_new_acc_test_norm))
    args.logger.info('Final Semisup Report Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
        img_all_acc_test, img_old_acc_test, img_new_acc_test))

    if img_all_acc_test_norm > best_all_acc_norm:
        best_all_acc_norm = img_all_acc_test_norm
        best_old_acc_norm = img_old_acc_test_norm
        best_new_acc_norm = img_new_acc_test_norm
        best_epoch_norm = epoch

        torch.save({
            'projection_head': projection_head.state_dict(),
            'vis_encoder': vis_encoder.state_dict(),
            'AL_model': AL_model.state_dict(),
            'text_encoder': text_encoder.state_dict()}
            , args.model_path[:-3] + f'_best.pt')
        print("Best model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

    if img_all_acc_test > best_all_acc:
        best_all_acc = img_all_acc_test
        best_old_acc = img_old_acc_test
        best_new_acc = img_new_acc_test
        best_epoch = epoch

    args.logger.info('Best Report          Epoch:{:.0f} Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f} | Eval_funcs: {}'.format(
            best_epoch_norm, best_all_acc_norm, best_old_acc_norm, best_new_acc_norm, args.test_eval_funcs))
    args.logger.info('Best Semisup Report  Epoch:{:.0f} Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f} | Eval_funcs: {}'.format(
            best_epoch, best_all_acc, best_old_acc, best_new_acc, args.test_eval_funcs))

    os.rename(args.log_dir, args.log_dir.replace('Acc','Acc-Final[%.1f_%.1f_%.1f]-Best[%.1f_%.1f_%.1f] ' % (
                                                     img_all_acc_test_norm * 100, img_old_acc_test_norm * 100,img_new_acc_test_norm * 100,
                                                     best_all_acc_norm * 100, best_old_acc_norm * 100,best_new_acc_norm * 100)))




def extract_labeled_protos(vis_encoder, text_encoder, AL_model, train_loader, args):
    vis_encoder.eval()
    AL_model.eval()
    text_encoder.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    ids=np.array([])
    mask_cls=np.array([])
    metrics=dict()
    text_feature_base = []
    text_features_labelled = text_encoder(text_tonkenized).to(device)
    for i in range(len(label_list)):
        text_feature_base.append(text_features_labelled[label_list[i]].unsqueeze(0))
    text_feature_base = torch.cat(text_feature_base, dim=0)

    for batch_idx, (images, label, uq_idx, mask_lab_) in enumerate(tqdm(train_loader)):

        images = images[0].to(device)
        label, mask_lab_ = label.to(device), mask_lab_.to(device).bool()
        images = images.type(clip_dtype).to(device)
        image_features = vis_encoder(images)
        text_features_AL = AL_model(image_features, vis_features_labelled, text_feature_base, args)
        vis_text_features = args.vis_rate * image_features + (1 - args.vis_rate) * text_features_AL

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = vis_text_features
        feats = feats.type(torch.float32).to(device)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        ids=np.append(ids,uq_idx.cpu().numpy())
        mask = np.append(mask, mask_lab_.cpu().bool().numpy())
        mask_cls = np.append(mask_cls,np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
    mask = mask.astype(bool)
    mask_cls = mask_cls.astype(bool)
    mask_cls_all = mask_cls
    # -----------------------
    # K-MEANS
    # -----------------------
    # args.logger.info('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = targets[mask]  # Get labelled targets
    u_targets = targets[~mask]  # Get unlabelled targets
    n_samples =len(targets)

    if args.unbalanced: cluster_size=None
    else: cluster_size=math.ceil(n_samples /(args.num_labeled_classes + args.num_unlabeled_classes))
    kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeanssem.labels_
    mask_cls=mask_cls[~mask]
    preds = all_preds.cpu().numpy()[~mask]


    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls,
                                                    eval_funcs=args.eval_funcs,
                                                    save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
    metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
    args.logger.info('SemiSupKMeans Clu Acc: All {:.4f} | Old {:.4f} | New {:.4f} | Eval_funcs: {}'.format(all_acc, old_acc, new_acc, args.eval_funcs))
    semi_sup_acc = all_acc

    prototype_higher=[]
    prototypes = kmeanssem.cluster_centers_
    prototype_higher.append(prototypes.clone())
    n_labeled=args.num_labeled_classes
    n_novel= args.num_unlabeled_classes
    label_proto = prototypes.cpu().numpy()[:args.num_labeled_classes,:]
    preds_higher=[]

    preds_higher.append(all_preds.clone())
    mask_known=(all_preds<args.num_labeled_classes).cpu().numpy()
    l_feats = all_feats[mask_known]  # Get labelled set
    u_feats = all_feats[~mask_known]
    l_feats, u_feats= (torch.from_numpy(x).to(device) for  x in (l_feats, u_feats))

    while n_labeled>1:
        n_labeled=max(int(n_labeled/2),1)
        n_novel=max(int(n_novel/2),1)

        kmeans_l = KMeans(n_clusters=n_labeled, random_state=0).fit(label_proto)
        preds_labels = torch.from_numpy(kmeans_l.labels_).to(device)
        level_l_targets=preds_labels[all_preds[mask_known]]
        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil( n_samples / (n_labeled+n_novel))
        kmeans_higher =SemiSupKMeans(k=n_labeled+n_novel, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None,cluster_size=cluster_size)
        kmeans_higher.fit_mix(u_feats, l_feats, level_l_targets)
        preds_level = kmeans_higher.labels_
        prototypes_level = kmeans_higher.cluster_centers_
        prototype_higher.append(prototypes_level.clone())
        preds_higher.append(preds_level.to(device).clone())

    return ids,all_preds, prototype_higher,preds_higher, metrics, semi_sup_acc


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class DRLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            target,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight

def train_AL (label_list, AL_model, vis_features_labelled, text_features, args, pre_train = False):
    tl = Averager()
    model = AL_model.train()
    DR_Loss = DRLoss()

    optimizer_al = torch.optim.SGD(model.parameters(), args.lr_AL, momentum=0.9, nesterov=True, weight_decay=args.weight_decay_AL)
    scheduler_al = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_al, T_max=args.epochs_AL, eta_min=args.lr * 0.01)
    cosine_loss = nn.CosineEmbeddingLoss(margin=0.1)
    args.logger.info('Analogical Learning stage:')

    for epoch in range(args.epochs_AL):
        
        class_list_decoer_new = sorted(random.sample(range(0, len(vis_features_labelled)), int(args.batch_size * 1)))
        class_list_decoer_base = list(set(range(0, len(vis_features_labelled))) - set(class_list_decoer_new))
        num_class_labelled = len(class_list_decoer_new)
        num_class_unlabelled = len(class_list_decoer_base)

        proto_list_base_f = torch.index_select(vis_features_labelled, 0, torch.tensor(class_list_decoer_base)).to(device).detach()
        proto_list_new_f = torch.index_select(vis_features_labelled, 0, torch.tensor(class_list_decoer_new)).to(device).detach()
        text_feature_base_idx = torch.index_select(label_list.to(device), 0, torch.tensor(class_list_decoer_base).to(device)).to(device)
        text_feature_new_idx = torch.index_select(label_list.to(device), 0, torch.tensor(class_list_decoer_new).to(device)).to(device)

        text_feature_base_f = []
        for i in range(len(text_feature_base_idx)):
            text_feature_base_f.append(text_features[text_feature_base_idx[i]].unsqueeze(0))

        text_feature_new_f = []
        for i in range(len(text_feature_new_idx)):
            text_feature_new_f.append(text_features[text_feature_new_idx[i]].unsqueeze(0))

        text_feature_base_f = torch.cat(text_feature_base_f, dim=0).detach()
        text_feature_new_f = torch.cat(text_feature_new_f, dim=0).detach()

        loss_flag = torch.ones([text_feature_new_f.shape[0]]).to(device)
        if pre_train and epoch < 5 : tqdm_gen = tqdm(range(args.epochs_AL_pre_train))
        else: tqdm_gen = tqdm(range(args.epochs_AL_pre))

        for i in enumerate(tqdm_gen):
            text_feature_new = model(proto_list_new_f, proto_list_base_f, text_feature_base_f, args)
            cosloss = cosine_loss(text_feature_new, text_feature_new_f, loss_flag) * 10
            text_feature_new = F.normalize(text_feature_new,dim=-1)
            text_feature_new_f_ = F.normalize(text_feature_new_f, dim=-1)
            drloss = DR_Loss (text_feature_new, text_feature_new_f_) * 10
            total_loss = drloss * 0 + cosloss * 1

            lrc = scheduler_al.get_last_lr()[0]
            tqdm_gen.set_description('AL training, epo {}, num_Labeled {}, num_Unlabeled {} lrc={:.4f}, ,COS loss={:.7f}, DR loss={:.7f},total loss={:.7f}'
                                     .format(epoch, num_class_labelled, num_class_unlabelled, lrc, cosloss.item(), drloss.item() ,total_loss.item()))
            tl.add(total_loss.item())

            optimizer_al.zero_grad()
            total_loss.backward()
            optimizer_al.step()

        scheduler_al.step()
        if epoch % 100 == 0:
            random.seed(epoch)
            args.logger.info('AL training, epo {}, num_Labeled {}, num_Unlabeled {} lrc={:.4f}, ,COS loss={:.7f}, DR loss={:.7f},total loss={:.7f}'
                                     .format(epoch, num_class_labelled, num_class_unlabelled, lrc, cosloss.item(), drloss.item() ,total_loss.item()))

    tl = tl.item()

    return tl

def get_proto_labelled(trainset_, model, args):
    model = model.eval()
    proto_loader_ = DataLoader(dataset=trainset_, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=False)
    embedding_list = []
    label_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(proto_loader_):
            images, class_labels, uq_idxs= batch
            class_labels = class_labels.to(device)
            images = torch.cat(images, dim=0)

            images = images.type(torch.float16).to(device)
            sup_labels = torch.cat([class_labels for _ in range(2)], dim=0)
            class_labels = sup_labels.to(device)

            embedding = model(images)
            embedding_list.append(embedding.cpu())
            label_list.append(class_labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    return embedding_list, label_list


def load_clip_to_cpu(cfg):
    backbone_name = cfg
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model



class TextEncoderClip(nn.Module):
    def __init__(self, clip_model , args, prompt_learn = False):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.prompt_learn = prompt_learn

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x



class Tokenized(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 0
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = ctx_vectors

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1','v2'])
    parser.add_argument('--test_eval_funcs', nargs='+', help='Which eval functions to test', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars, aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    parser.add_argument('--strategy', type=str, default='zero_one')
    parser.add_argument('--cluster_momentum', type=float, default=1)

    parser.add_argument('--unsupervised_smoothing', type=float, default=1)
    parser.add_argument('--distance', type=str, default='euclidean', help='options: euclidean, cosine')
    parser.add_argument('--prototype_extraction_interval', default=1, type=int)
    parser.add_argument('--gpu_clustering', type=str2bool, default=True)
    parser.add_argument('--unbalanced', type=str2bool, default=False)
    parser.add_argument('--report', type=str2bool, default=True)


    parser.add_argument('--lr_AL', type=float, default=0.1)
    parser.add_argument('--gamma_AL', type=float, default=0.1)
    parser.add_argument('--epochs_AL', default=5000, type=int)
    parser.add_argument('--epochs_AL_pre', default=10, type=int)
    parser.add_argument('--epochs_AL_pre_train', default=2000, type=int)
    parser.add_argument('--weight_decay_AL', type=float, default=1e-4)
    parser.add_argument('--step_AL', type=int, default=30)
    parser.add_argument('--al_loss_rate', type=float, default=0)
    parser.add_argument('--layers_AL', default=2, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--vis_rate', default=0.5, type=float)
    parser.add_argument('--train_report_interval', default=40, type=int)


    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:%d' % args.gpu_id)
    args = get_class_splits(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.interpolation = 3
    args.crop_pct = 0.875

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['ALGCD'])
    args.logger.info(f'Using evaluation function {args.test_eval_funcs} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------

    clip_model = load_clip_to_cpu(cfg="ViT-B/16")
    vis_encoder = clip_model.visual.to(device)
    text_encoder = TextEncoderClip(clip_model, args).to(device)
    AL_model = ALNetCLIP().to(device)
    clip_dtype = torch.float16

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 512
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536


    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets, classesname, unlabelled_train_all_test = get_datasets(
                                                                                                                        args.dataset_name,
                                                                                                                        train_transform,
                                                                                                                        test_transform,
                                                                                                                        args)

    # --------------------
    # get prototype and prompt
    # --------------------
    classesname_labelled = classesname[0: args.num_labeled_classes]

    text = []
    for name in classesname_labelled:
        text_base = "a photo of a XXX"
        if args.dataset_name == 'scars':
            text_get = text_base.replace("XXX", str(name[0]))
        elif args.dataset_name == 'aircraft':
            text_get = text_base.replace("XXX", str(name).replace("\n", ""))
        else:
            text_get = text_base.replace("XXX", str(name))
        text.append(text_get)
        args.logger.info("Use for AL_train:" + text_get)

    text_tonkenized = clip.tokenize(text).to(device)
    vis_features_labelled, label_list = get_proto_labelled(train_dataset.labelled_dataset, vis_encoder, args)
    text_features = text_encoder(text_tonkenized).to(device)

    # --------------------
    # Anological Learning
    # --------------------

    train_AL(label_list, AL_model, vis_features_labelled, text_features, args, pre_train=True)
    vis_features_labelled = vis_features_labelled.to(device)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / (unlabelled_len+label_len) for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    merge_train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(unlabelled_train_all_test, num_workers=args.num_workers,batch_size=args.batch_size, shuffle=False)


    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in vis_encoder.parameters():
        m.requires_grad = False
    max_block = 0
    for name, m in vis_encoder.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[2])
            if block_num > max_block:
                max_block = block_num
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    for m in text_encoder.parameters():
        m.requires_grad = False
    max_block = 0
    for name, m in text_encoder.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[2])
            if block_num > max_block:
                max_block = block_num
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    for name, m in AL_model.named_parameters():
        m.requires_grad = False

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    if args.warmup_model_dir is not None:
        print(f'Loading projection head weights from {args.warmup_model_dir}')
        projection_head.load_state_dict(
            torch.load(args.warmup_model_dir + 'model_proj_head_best.pt', map_location='cpu'), strict=False)

    projection_head.to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    if not os.path.exists('Plots'):
        os.mkdir('Plots')
    train(projection_head, vis_encoder, text_encoder, AL_model, train_loader, test_loader_labelled, merge_train_loader, args)

