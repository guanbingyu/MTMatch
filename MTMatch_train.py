import os
import argparse
import json
import sys
import numpy as np
import random
import torch
import mlflow
from pathlib import Path

from selfsl.dataset import DMDataset
from selfsl.path_utils import resolve_task_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="DBLP-ACM")
    parser.add_argument("--task_type", type=str, default="em")
    parser.add_argument("--logdir", type=str, default="outputs/checkpoints")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument(
        "--em_head_type",
        type=str,
        default="full",
        choices=["full", "paired_only"],
        help="Record-level EM head type.",
    )
    parser.add_argument("--fp16", dest="fp16", action="store_true", help="使用自动混合精度 (推荐 50 系显卡开启)")

    # ssl related
    parser.add_argument("--ssl_method", type=str, default="simclr")
    parser.add_argument("--n_ssl_epochs", type=int, default=0)

    # data augmentation
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--cutoff_ratio", type=float, default=0.05)

    # negative sampling
    parser.add_argument("--clustering", dest="clustering", action="store_true")
    parser.add_argument("--num_clusters", type=int, default=90)

    # bootstraping related
    parser.add_argument("--zero", dest="zero", action="store_true")
    parser.add_argument("--bootstrap", dest="bootstrap", action="store_true")
    parser.add_argument("--multiplier", type=int, default=8)

    # barlow twins
    parser.add_argument('--projector', default='768', type=str,
                        metavar='MLP', help='projector MLP')
    parser.add_argument('--scale-loss', default=1.0/256, type=float,
                        metavar='S', help='scale the loss')
    parser.add_argument('--lambd', default=3.9e-3, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--alpha_bt', default=0.001, type=float,
                        help='weight of the BT loss')

    # re-use checkpoints
    parser.add_argument("--save_ckpt", dest="save_ckpt", action="store_true",
                        help='save the ssl checkpoint')
    parser.add_argument("--use_saved_ckpt", dest="use_saved_ckpt", action="store_true",
                        help='use the saved ssl checkpoint if available')
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from a saved checkpoint such as best_model.pt.",
    )

    # for blocking
    parser.add_argument("--blocking", dest="blocking", action="store_true",
                        help="if set, evaluate blocking during training")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tfidf", dest="tfidf", action="store_true",
                        help="if set, apply the baseline blocker")
    parser.add_argument("--lm_only", dest="lm_only", action="store_true",
                        help="if set, only apply a non-fine-tuned LM")
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="f1",
        choices=["f1", "fbeta", "precision"],
        help="Validation objective used for threshold selection.",
    )
    parser.add_argument(
        "--selection_beta",
        type=float,
        default=1.0,
        help="Beta used when selection_metric=fbeta.",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.05,
        help="Search step for threshold tuning on the validation set.",
    )
    parser.add_argument(
        "--neg_class_weight",
        type=float,
        default=1.0,
        help="Class weight for the negative class during supervised fine-tuning.",
    )
    parser.add_argument(
        "--pos_class_weight",
        type=float,
        default=1.0,
        help="Class weight for the positive class during supervised fine-tuning.",
    )

    # mlflow tag
    parser.add_argument("--mlflow_tag", type=str, default=None)

    hp = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    tracking_dir = package_root / "outputs" / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    package_root = Path(__file__).resolve().parent
    tracking_dir = package_root / "outputs" / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    # mlflow logging
    for variable in ["task", "batch_size", "lr", "n_epochs",
                     "ssl_method", "alpha_bt",
                     "da", "cutoff_ratio",
                     "clustering", "num_clusters",
                     "n_ssl_epochs",
                     "zero", "bootstrap", "multiplier",
                     "em_head_type",
                     "selection_metric", "selection_beta", "threshold_step",
                     "neg_class_weight", "pos_class_weight"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)

    # set seed & GPU optimization
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # --- 针对 50 系列显卡的 Blackwell 架构优化 ---
        # 允许 TensorCore 使用 TF32 精度，极大提升训练速度且几乎不损精度
        torch.set_float32_matmul_precision('high') 
        # 自动选择最快卷积算法
        torch.backends.cudnn.benchmark = True

    path = resolve_task_dir(hp.task_type, hp.task)
    train_path_nolabel = os.path.join(path, 'train_no_label.txt')

    train_path = os.path.join(path, 'train.txt')
    valid_path = os.path.join(path, 'valid.txt')
    if not os.path.exists(valid_path):
        valid_path = os.path.join(path, 'train.txt')

    test_path = os.path.join(path, 'test.txt')

    trainset = DMDataset(train_path,
                         lm=hp.lm,
                         size=hp.size,
                         max_len=hp.max_len,
                         da=None) # data augmentation
    validset = DMDataset(valid_path,
                         lm=hp.lm,
                         size=None,
                         max_len=hp.max_len)
    testset = DMDataset(test_path,
                        lm=hp.lm,
                        size=None,
                        max_len=hp.max_len)

    if hp.ssl_method in ['barlow_twins', 'simclr', 'combined'] \
            and hp.n_ssl_epochs > 0:
        method = hp.ssl_method
    else:
        method = 'ditto'

    if hp.zero:
        method += '_zero'
    if hp.bootstrap:
        method += '_bootstrap'
    if hp.clustering:
        method += '_clustering'

    method += '_' + str(hp.da)

    run_tag = '%s_%s_da=%s_id=%d_size=%s' % (hp.task_type, hp.task, method, hp.run_id, str(hp.size))

    if hp.ssl_method in ['barlow_twins', 'simclr', 'combined']:
        # self-supervised learning
        # 此时会调用已经移除 Apex 并适配了 torch.amp 的 train 函数
        from selfsl.barlow_twins_simclr import train
        from selfsl.bt_dataset import BTDataset

        trainset_nolabel = BTDataset(train_path_nolabel,
                             lm=hp.lm,
                             size=10000,
                             max_len=hp.max_len,
                             da=hp.da) # data augmentation

        trainset_nolabel.create_ground_truth([DMDataset(path, lm=hp.lm, max_len=hp.max_len, size=None) \
                                              for path in [train_path, valid_path, test_path]])
        train(trainset_nolabel, trainset, validset, testset, run_tag, hp)
    else:
        # Ditto 分支
        from selfsl.dm import train
        train(trainset, validset, testset, run_tag, hp)
