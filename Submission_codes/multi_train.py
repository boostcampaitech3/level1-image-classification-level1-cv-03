import argparse
from collections import defaultdict
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import wandb
import pdb

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    if(args.model_load.lower() == "true"):
        model_path = os.path.join(os.path.join(model_dir, args.name), 'best.pth')
        model.load_state_dict(torch.load(model_path))
    model = torch.nn.DataParallel(model)


    leraning_book = ["mask","age","gender"]
    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = {
        "mask":opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        ),
        "age":opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        ),
        "gender":opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
    }
    scheduler={}
    for k in leraning_book:
        scheduler[k] = StepLR(optimizer[k], args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = defaultdict(int)
    best_val_loss = defaultdict(lambda:float('inf'))
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = defaultdict(int)
        matches = defaultdict(int)
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels#.to(device)

            for k in leraning_book:
                optimizer[k].zero_grad()
            loss = {}
            preds = {}
            outs = model(inputs)

            for k,v in outs.items():
                if k in leraning_book:
                    preds[k] = torch.argmax(outs[k], dim=-1)
                    loss[k] = criterion(outs[k], labels[k].to(device))
                    loss[k].backward()
                    optimizer[k].step()

            for k,v in loss.items():
                if k in leraning_book:
                    loss_value[k] += loss[k].item()
                    matches[k] += torch.sum(preds[k] == labels[k].data.to(device))
            
            train_loss = {}
            train_acc = {}
            # 한 epoch이 모두 종료되었을 때,
            for k in leraning_book:
                if (idx + 1) % args.log_interval == 0:
                    print(k+" Layer Status-")
                    train_loss[k] = loss_value[k] / args.log_interval
                    train_acc[k] = matches[k] / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer[k])
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss[k]:4.4} || training accuracy {train_acc[k]:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss" + k, train_loss[k], epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy" + k, train_acc[k], epoch * len(train_loader) + idx)
                    wandb.log({
                        "Train Accuracy" + k: train_acc[k],
                        "Train Loss" + k: train_loss[k]
                    })
                    loss_value[k] = 0
                    matches[k] = 0
        for k in leraning_book:
            scheduler[k].step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = defaultdict(list)
            val_acc_items = defaultdict(list)
            figure = {i:None for i in leraning_book}
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels#.to(device)

                outs = model(inputs)
                loss_value=defaultdict(int)
                loss_item=defaultdict(int)
                acc_item=defaultdict(int)
                for k,v in outs.items():
                    if k in leraning_book:
                        preds[k] = torch.argmax(outs[k], dim=-1)
                        loss_value[k] += loss[k].item()
                        loss_item[k] = criterion(outs[k], labels[k].to(device)).item()
                        acc_item[k] = (preds[k] == labels[k].data.to(device)).sum().item()
                        val_loss_items[k].append(loss_item[k])
                        val_acc_items[k].append(acc_item[k])

                        if figure[k] is None:
                            inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                            inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                            figure[k] = grid_image(
                                inputs_np, labels[k], preds[k], n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                            )

            val_loss={}
            val_acc={}
            for k in leraning_book:
                print(k+" Layer Status-")
                val_loss[k] = np.sum(val_loss_items[k]) / len(val_loader)
                val_acc[k] = np.sum(val_acc_items[k]) / len(val_set)
                best_val_loss[k] = min(best_val_loss[k], val_loss[k])
                if val_acc[k] > best_val_acc[k]:
                    print(f"New best model for val accuracy : {val_acc[k]:4.2%}! saving the best model..")
                    if k=="mask":
                        torch.save(model.module.mask.state_dict(), f"{save_dir}/mask_best.pth")
                    elif k=="age":
                        torch.save(model.module.age.state_dict(), f"{save_dir}/age_best.pth")
                    elif k=="gender":
                        torch.save(model.module.gender.state_dict(), f"{save_dir}/gender_best.pth")
                    best_val_acc[k] = val_acc[k]
                
                if k=="mask":
                    torch.save(model.module.mask.state_dict(), f"{save_dir}/mask_last.pth")
                elif k=="age":
                    torch.save(model.module.age.state_dict(), f"{save_dir}/age_last.pth")
                elif k=="gender":
                    torch.save(model.module.gender.state_dict(), f"{save_dir}/gender_last.pth")
                print(
                    f"[Val] acc : {val_acc[k]:4.2%}, loss: {val_loss[k]:4.2} || "
                    f"best acc : {best_val_acc[k]:4.2%}, best loss: {best_val_loss[k]:4.2}"
                )

                logger.add_scalar("Val/loss" + k, val_loss[k], epoch)
                logger.add_scalar("Val/accuracy" + k, val_acc[k], epoch)
                logger.add_figure("results" + k, figure[k], epoch)
                wandb.log({
                        "Test Accuracy" + k: val_acc[k],
                        "Test Loss" + k: val_loss[k]
                })
                wandb.log({"chart" + k: figure[k]})
                print()

if __name__ == '__main__':

    # #wandb.config.epochs = 4

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
    #                     help='input batch size for training (default: 8)')
    # args = parser.parse_args()
    

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--model_load', type=str, default="False")

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/backup/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    
    

    args = parser.parse_args()

    wandb.init(project="project-name", reinit=True)
    wandb.run.name = increment_path(os.path.join(args.model_dir, args.name))
    wandb.run.save()
    wandb.config.update(args) # adds all of the arguments as config variables
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
