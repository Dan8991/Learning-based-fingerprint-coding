from model import MeanScaleHyperpriorGrayscale
from torchvision import transforms
from dataset import FingerprintTraining, FinetuningDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import argparse
from compressai.zoo import mbt2018_mean as msh
from compressai.zoo.image import cfgs

parser = argparse.ArgumentParser()
parser.add_argument("--q", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--fine_tuning", action="store_true")
parser.add_argument("--dataset_path", type=str)

qs = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932, 0.18]

args = parser.parse_args()

if args.fine_tuning:
    model = msh(quality=args.q, pretrained=True)
    train_dataset = FinetuningDataset(
        args.dataset_path,
        transform=transforms.Compose(
            [transforms.Resize(320),
            transforms.CenterCrop((320, 320)),
            transforms.ToTensor()]),
    )
else:
    model = MeanScaleHyperpriorGrayscale(*cfgs["mbt2018-mean"][args.q])
    train_dataset = FingerprintTraining(
        os.path.join(args.dataset_path, "casia", "original"),
        transform=transforms.Compose([
            transforms.CenterCrop((320, 320)),
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor(),
        ]), split="train")
    
val_dataset = FingerprintTraining(
    os.path.join(args.dataset_path, "casia", "original"),
    transform=transforms.Compose([
        transforms.CenterCrop((320, 320)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]), split="val")

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)

epochs = 1000

parameters = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
aux_parameters = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]

optimizer = torch.optim.Adam(parameters, lr=args.lr)
aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

criterion = torch.nn.MSELoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.train()

best_loss = float("inf")

for epoch in range(epochs):
    
    model.train()
    
    losses = []
    rates = []
    dists = []
    aux_losses = []

    for img in tqdm(train_dataloader):

        img = img.to(device)

        comp = model(img)
        recon = comp["x_hat"]

        dist = criterion(recon, img)
        rate = -torch.log2(comp["likelihoods"]["y"]).mean() - torch.log2(comp["likelihoods"]["z"]).mean()
        loss = dist * 255 ** 2 * qs[args.q - 1] + rate
        
        aux_loss = model.entropy_bottleneck.loss()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        aux_optimizer.zero_grad()
        aux_loss.backward()
        aux_optimizer.step()
    
    model.eval()
    with torch.no_grad():
        
        val_loss = 0
        val_dist = 0
        aux_loss = 0
        val_rate = 0

        i = 0

        for img in tqdm(val_dataloader):
            i += 1
            img = img.to(device)
            
            if args.fine_tuning:
                img = img.repeat(1, 3, 1, 1)

            comp = model(img)
            recon = comp["x_hat"]
            
            dist = criterion(recon, img)
            rate = -torch.log2(comp["likelihoods"]["y"]).mean() - torch.log2(comp["likelihoods"]["z"]).mean()
            loss = dist * 255 ** 2 * qs[args.q - 1] + rate
            
            val_loss += loss.item()

        scheduler.step(val_loss/i)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join("..", "models", "training_models" if not args.fine_tuning else "finetuned_models", f"ms-hyperprior-{args.q}.pt"))
            print(f"Epoch {epoch} - Val loss: {val_loss/len(val_dataloader)} - Saving model")

        print(f"Epoch {epoch} - Val loss: {val_loss/len(val_dataloader)}")

    if optimizer.param_groups[0]["lr"] < 1e-6:
        quit()
