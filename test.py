# test_from_logdir.py
# Evaluate BOTH best and final checkpoints automatically inferred from a log directory.
#
# Example:
#   python test_from_logdir.py --dataset_name cub --log_dir /path/to/your/log_dir --gpu_id 0
#
# Notes:
# - This script searches ckpt files in log_dir recursively.
# - It prefers filenames containing "best" and "final".
# - It then runs evaluation twice: (1) best ckpt, (2) final ckpt.

import argparse
import sys
import inspect
import time
from pathlib import Path
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
from project_utils.cluster_utils import mixed_eval, AverageMeter
from loguru import logger
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits
from config import exp_root, dino_pretrain_path, dino_pretrain_path2
from datetime import datetime
from project_utils.general_utils import  get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm

from loguru import logger
from project_utils.cluster_and_log_utils import log_accs_from_preds

from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans


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
        if args.proto_based:
            log_dir = os.path.join(root_dir, "Proto_Based", args.dataset_name + "-Acc" + now)
        else:
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

# -------------------------
# Helper: checkpoint discovery
# -------------------------
def _score_ckpt_name(name: str, want: str) -> int:
    """
    Higher score = better match.
    want: "best" or "final"
    """
    n = name.lower()
    score = 0

    # Strong signals
    if want in n:
        score += 100
    if f"_{want}." in n or f"-{want}." in n:
        score += 50

    # Common training naming patterns
    if "model" in n:
        score += 10
    if "ckpt" in n or "checkpoint" in n:
        score += 10

    # Prefer .pt/.pth
    if n.endswith(".pt") or n.endswith(".pth"):
        score += 10

    # Penalize obvious mismatches
    other = "final" if want == "best" else "best"
    if other in n:
        score -= 30

    return score


def infer_checkpoints_from_log_dir(log_dir: str):
    """
    Recursively search log_dir for checkpoint files.
    Return: (best_path, final_path)
    """
    p = Path(log_dir)
    assert p.exists() and p.is_dir(), f"log_dir not found or not a directory: {log_dir}"

    # Collect candidate files
    cand = []
    for ext in ("*.pt", "*.pth"):
        cand.extend(list(p.rglob(ext)))

    if not cand:
        raise FileNotFoundError(f"No .pt/.pth found under log_dir: {log_dir}")

    # Rank candidates for "best" and "final"
    best_ranked = sorted(cand, key=lambda x: _score_ckpt_name(x.name, "best"), reverse=True)
    final_ranked = sorted(cand, key=lambda x: _score_ckpt_name(x.name, "final"), reverse=True)

    best_path = best_ranked[0]
    final_path = final_ranked[0]

    # If they end up being the same file (e.g., only one ckpt exists), try to pick a different one
    if best_path.resolve() == final_path.resolve() and len(cand) >= 2:
        # pick next best for final
        for x in final_ranked[1:]:
            if x.resolve() != best_path.resolve():
                final_path = x
                break

    return str(best_path), str(final_path)


# -------------------------
# Clustering / evaluation (kept consistent with your training-time logic)
# -------------------------


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


def test(vis_encoder, text_encoder, AL_model, test_loader, args, epoch):
    """
    Runs evaluation once. The following globals are prepared in main():
      - device
      - clip_dtype
      - text_tonkenized
      - label_list
      - vis_features_labelled
    """
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
            image_features = vis_encoder(images).to(device)

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

    img_all_acc_test_norm, img_old_acc_test_norm, img_new_acc_test_norm = test_agglo_all(mask_test, epoch,all_feats_val_norm, targets, mask, args)
    args.logger.info('Know-K       Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f} | Eval_funcs: {}'.format(
        img_all_acc_test_norm, img_old_acc_test_norm, img_new_acc_test_norm, args.test_eval_funcs))

    img_all_acc_test, img_old_acc_test, img_new_acc_test = test_agglo_semisup(mask_test, epoch, all_feats_val_norm,targets, mask, args)
    args.logger.info('Know-K  SemiSup   Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f} | Eval_funcs: {}'.format(
        img_all_acc_test, img_old_acc_test, img_new_acc_test, args.test_eval_funcs))
    return  img_all_acc_test_norm, img_old_acc_test_norm, img_new_acc_test_norm



# -------------------------
# CLIP helpers (kept aligned with your training code)
# -------------------------
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
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class ContrastiveLearningViewGenerator(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


def get_sample_labelled(trainset_, model, args):
    model = model.eval()
    proto_loader_ = DataLoader(dataset=trainset_, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=False)
    embedding_list = []
    label_list = []

    # data_list=[]
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


def get_proto_labelled(trainset_, model, args):
    model = model.eval()
    proto_loader_ = DataLoader(dataset=trainset_, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=False)
    embedding_list = []
    label_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(proto_loader_):
            images, class_labels, uq_idxs = batch
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

    proto_list = []
    label_list_proto = []
    for class_index in range(args.num_labeled_classes):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
        label_list_proto.append(class_index)

    proto_list = torch.stack(proto_list, dim=0)
    label_list_proto = torch.tensor(label_list_proto)

    return proto_list, label_list_proto


def load_checkpoint_into_models(ckpt_path, projection_head, vis_encoder, text_encoder, AL_model):
    """
    Load a torch.save(dict-of-state_dicts) checkpoint into model components.
    Works even if some keys are missing (strict=False).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    missing = []
    for k, m in [
        ("projection_head", projection_head),
        ("vis_encoder", vis_encoder),
        ("text_encoder", text_encoder),
        ("AL_model", AL_model),
    ]:
        if k not in ckpt:
            missing.append(k)
            continue
        m.load_state_dict(ckpt[k], strict=False)

    if missing:
        logger.warning(f"Checkpoint missing keys: {missing} (will continue evaluation).")

    return ckpt


# -------------------------
# Main
# -------------------------
def run_one_ckpt(ckpt_path: str, tag: str, args, vis_encoder, text_encoder, AL_model, projection_head,
                 test_loader):
    """
    Load a checkpoint, rebuild prototypes (since vis_encoder may change), then evaluate.
    """
    logger.info(f"========== [{tag}] Loading checkpoint ==========")
    logger.info(f"[{tag}] ckpt = {ckpt_path}")
    load_checkpoint_into_models(ckpt_path, projection_head, vis_encoder, text_encoder, AL_model)

    logger.info(f"========== [{tag}] Evaluating ==========")
    img_all_acc_test, img_old_acc_test, img_new_acc_test = test(vis_encoder, text_encoder, AL_model, test_loader, args, epoch=0)
    logger.info(f"========== [{tag}] Done ==========")
    return img_all_acc_test, img_old_acc_test, img_new_acc_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prop_train_labels", type=float, default=0.5)
    parser.add_argument("--use_ssb_splits", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # Evaluation hooks used by your logger utility
    parser.add_argument("--eval_funcs", nargs="+", default=["v2"])
    parser.add_argument("--test_eval_funcs", nargs="+", default=["v2"])

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--transform", type=str, default="imagenet")
    parser.add_argument('--exp_root', type=str, default=exp_root)

    # Model/runtime params
    parser.add_argument("--dataset_name", type=str, default="cub")
    parser.add_argument("--n_views", type=int, default=2)
    parser.add_argument("--vis_rate", type=float, default=0.5)
    parser.add_argument("--unbalanced", type=str2bool, default=False)
    parser.add_argument('--layers_AL', default=2, type=int)
    parser.add_argument("--proto_based", type=str2bool, default=True)

    # Note: You can specify any log_dir here that contains ckpt files. The script will try to infer which ckpt is "best" vs "final" based on filename patterns.
    parser.add_argument("--ckpt_dir", type=str, default="LLogs/ALGCD/Proto_Based/cub-Acc-Final[84.4_79.5_86.9]-Best[84.9_79.9_87.4]_Final_Semi[83.4_79.4_85.4]-Best_Semi[84.7_79.3_87.3] (08.02.2026_|_42.143)/checkpoints")


    args = parser.parse_args()
    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    init_experiment(args, runner_name=['ALGCD_test'])
    args.logger.info(f'Using evaluation function {args.test_eval_funcs} to print results')

    # Global runtime settings (kept consistent)
    global device, clip_dtype
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    clip_dtype = torch.float16

    # Resolve class splits
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # Infer both checkpoints from log_dir
    best_ckpt, final_ckpt = infer_checkpoints_from_log_dir(args.ckpt_dir)
    logger.info(f"Inferred checkpoints:\n  best : {best_ckpt}\n  final: {final_ckpt}")

    # Build models (same structure as training)
    clip_model = load_clip_to_cpu(cfg="ViT-B/16")
    vis_encoder = clip_model.visual.to(device)
    text_encoder = TextEncoderClip(clip_model).to(device)
    AL_model = ALNetCLIP().to(device)

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 512
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536
    args.interpolation = 3
    args.crop_pct = 0.875

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
        #args.logger.info("Use for AL_Test:" + text_get)

    global vis_features_labelled, label_list,text_tonkenized
    text_tonkenized = clip.tokenize(text).to(device)
    if args.proto_based:
        vis_features_labelled, label_list = get_proto_labelled(train_dataset.labelled_dataset, vis_encoder, args)
    else:
        vis_features_labelled, label_list = get_sample_labelled(train_dataset.labelled_dataset, vis_encoder, args)
    vis_features_labelled = vis_features_labelled.to(device)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projection_head.to(device)


    test_loader = DataLoader(
        unlabelled_train_all_test,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Run best
    img_all_acc_test_best, img_old_acc_test_best, img_new_acc_test_best = run_one_ckpt(
        ckpt_path=best_ckpt,
        tag="BEST",
        args=args,
        vis_encoder=vis_encoder,
        text_encoder=text_encoder,
        AL_model=AL_model,
        projection_head=projection_head,
        test_loader=test_loader
    )

    # Run final
    img_all_acc_test, img_old_acc_test, img_new_acc_test = run_one_ckpt(
        ckpt_path=final_ckpt,
        tag="FINAL",
        args=args,
        vis_encoder=vis_encoder,
        text_encoder=text_encoder,
        AL_model=AL_model,
        projection_head=projection_head,
        test_loader=test_loader,
    )

    os.rename(args.log_dir, args.log_dir.replace('Acc', 'Acc-Final[%.2f_%.2f_%.2f]-Best[%.2f_%.2f_%.2f] ' % (
        img_all_acc_test * 100, img_old_acc_test * 100, img_new_acc_test * 100,
        img_all_acc_test_best * 100, img_old_acc_test_best * 100 , img_new_acc_test_best * 100)))

if __name__ == "__main__":
    main()
