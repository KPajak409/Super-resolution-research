import argparse
import os
import copy
import math
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import wandb
from src.models.FSRCNN import FSRCNN
from src.models.ESPCN import espcn_x2, espcn_x3, espcn_x4
from src.datasets.train_dataset import TrainDataset, EvalDataset
from src.utils import AverageMeter
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--outputs-dir", type=str, required=True)
    parser.add_argument("--weights-file", type=str)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="super-resolution-research",
        # track hyperparameters and run metadata
        config={
            "architecture": "espcnx2",
            "dataset": "T91_patches",
            # "lr_scheduler": "ReduceLRonPlateau",
            "img_hr_size": 20,
            "scale": args.scale,
            # "ds_split": {
            #     "train": 0.9,
            #     "test": 0.0,
            #     "val": 0.1,
            #     "random_seed": 42,
            # },
            "hyperparams": {
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "seed": args.seed,
                "num_workers": args.num_workers,
            },
        },
        mode="online",
    )

    args.outputs_dir = os.path.join(args.outputs_dir, "x{}".format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    model = espcn_x2(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [
            {"params": model.conv1.parameters()},
            {"params": model.conv2.parameters()},
            {"params": model.conv3.parameters(), "lr": args.lr * 0.1},
        ],
        lr=args.lr,
    )
    # model = FSRCNN(scale_factor=args.scale).to(device)
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(
    #     [
    #         {"params": model.first_part.parameters()},
    #         {"params": model.mid_part.parameters()},
    #         {"params": model.last_part.parameters(), "lr": args.lr * 0.1},
    #     ],
    #     lr=args.lr,
    # )

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    total_iterations = math.ceil(len(train_dataset) / args.batch_size)
    log_iteration = math.floor(total_iterations * 0.1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        epoch_psnr = AverageMeter()
        with tqdm(
            total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80
        ) as t:
            t.set_description("epoch: {}/{}".format(epoch + 1, args.num_epochs))

            for i, data in enumerate(train_dataloader):
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=f"{epoch_losses.avg:.6f}")
                t.update(len(inputs))

                if (i + 1) % log_iteration == 0:
                    psnr_value = psnr(preds, labels, (0.0, 1.0)).item()
                    ssim_value = ssim(preds, labels, (0.0, 1.0)).item()
                    run.log(
                        {
                            "train/loss": epoch_losses.val,
                            "train/psnr": psnr_value,
                            "train/ssim": ssim_value,
                        }
                    )

        torch.save(
            model.state_dict(),
            os.path.join(args.outputs_dir, "epoch_{}.pth".format(epoch)),
        )

        model.eval()
        epoch_psnr_val = AverageMeter()
        epoch_ssim_val = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr_val.update(psnr(preds, labels, (0.0, 1.0)).item(), len(inputs))
            epoch_ssim_val.update(ssim(preds, labels, (0.0, 1.0)).item(), len(inputs))
        run.log(
            {
                "val/psnr": epoch_psnr_val.avg,
                "val/ssim": epoch_psnr_val.avg,
            }
        )
        print(f"eval psnr: {epoch_psnr_val.avg:.2f} ssim: {epoch_ssim_val.avg:.3f}")

        if epoch_psnr_val.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr_val.avg
            best_weights = copy.deepcopy(model.state_dict())

    print("best epoch: {}, psnr: {:.2f}".format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, "best.pth"))
