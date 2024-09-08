# %%
import torch
import math
import wandb
import torchvision.transforms.v2 as transforms
import torchmetrics
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from ImageNet_dataset import ImageNet
from conv_tests import Conv_upsamplex2, Conv_upsamplex4, Conv_upsamplex2_tanh
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

print(
    f"""cuda_is_available: {torch.cuda.is_available()}
device_count: {torch.cuda.device_count()}
current_device: {torch.cuda.current_device()}"""
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# start a new wandb run to track this script
run = wandb.init(
    # set the wandb project where this run will be logged
    project="super-resolution-research",
    # track hyperparameters and run metadata
    config={
        "architecture": "ESPCNx2_lower",
        "dataset": "ImageNet1k",
        "lr_scheduler": "ReduceLRonPlateau",
        "ds_split": {"train": 0.3, "val": 0.7, "test": None, "random_seed": 42},
        "hyperparams": {"learning_rate": 0.0003, "batch_size": 512, "epochs": 20},
    },
    mode="online",
)

hyperparams = run.config["hyperparams"]

# %%
composed = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomVerticalFlip(p=0.2),
    ]
)
ds_lrx4 = ImageNet(
    input_folder_path="./data/test/LRx4",
    target_folder_path="./data/test/LRx2",
    transform=composed,
)

generator1 = torch.Generator().manual_seed(run.config["ds_split"]["random_seed"])
valtrain_ds_list = random_split(
    ds_lrx4,
    [run.config["ds_split"]["train"], run.config["ds_split"]["val"]],
    generator=generator1,
)

train_dl = DataLoader(
    dataset=valtrain_ds_list[0],
    batch_size=hyperparams["batch_size"],
    shuffle=True,
)
val_dl = DataLoader(
    dataset=valtrain_ds_list[1],
    batch_size=hyperparams["batch_size"],
    shuffle=True,
)

model = Conv_upsamplex2().to(device=device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", factor=0.8, patience=2
)
total_iterations = math.ceil(len(valtrain_ds_list[0]) / hyperparams["batch_size"])
log_iteration = math.floor(total_iterations * 0.1)

# %%
losses = []

for epoch_index in range(hyperparams["epochs"]):
    with tqdm(train_dl) as tepoch:
        tepoch.set_description(f"Epoch: {epoch_index+1}")

        for i, (inputs, targets) in enumerate(tepoch):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()
            losses.append(loss.item())
            if (i + 1) % log_iteration == 0:
                loss_value = loss.item()
                psnr_value = psnr(outputs, targets).item()
                ssim_value = ssim(outputs, targets).item()

                my_lr = scheduler.optimizer.param_groups[0]["lr"]
                run.log(
                    {
                        "train/loss": loss_value,
                        "train/psnr": psnr_value,
                        "train/ssim": ssim_value,
                        "lr": my_lr,
                    }
                )
                tepoch.set_postfix(
                    lr=f"{round(my_lr, 7)}", psnr=psnr_value, ssim=ssim_value
                )
        scheduler.step(sum(losses) / len(losses))
        losses = []
# %%
run.finish()
# %%
torch.save(model, "./models/ESPCNx4")
