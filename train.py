# %%
import torch
import math
import wandb
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision.transforms.v2 as transforms
import torchvision.utils as utils
from ImageNet_dataset import ImageNet
from conv_tests import Conv_upsamplex2

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
        "architecture": "CNN",
        "dataset": "ImageNet1k",
        "ds_split": {"train": 0.9, "val": 0.1, "test": None, "random_seed": 42},
        "hyperparams": {"learning_rate": 0.001, "batch_size": 2048, "epochs": 10},
    },
    mode="online",
)

hyperparams = run.config["hyperparams"]

# %%
composed = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.1),
    ]
)
ds_lrx4 = ImageNet("./data/test/LRx4", "./data/test/LRx2", transform=composed)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
total_iterations = math.ceil(len(valtrain_ds_list[0]) / hyperparams["batch_size"])
log_iteration = math.floor(total_iterations * 0.1)
# %%
running_loss = 0

for epoch_index in range(hyperparams["epochs"]):
    last_loss = 0

    for i, (inputs, targets) in enumerate(tqdm(train_dl)):
        optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % log_iteration == 0:
            last_loss = running_loss / log_iteration  # loss per batch
            # print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dl) + i + 1
            run.log({"train/loss": last_loss})
            running_loss = 0.0

# %%
run.finish()

# %%
torch.save(model, "./models/upsample2x_test")

# %%
