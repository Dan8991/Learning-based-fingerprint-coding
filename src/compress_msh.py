from compressai.zoo import mbt2018_mean as msh
from torchvision import transforms
from dataset import Fingerprint
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
from tqdm import tqdm
from model import MeanScaleHyperpriorGrayscale
from compressai.zoo.image import cfgs
import pickle
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--fine_tuned", action="store_true")
args = parser.parse_args()
orig_path = args.path
shape = (320, 320)
dataset = Fingerprint(orig_path, transform=transforms.Compose([transforms.CenterCrop(shape), transforms.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=6)
qualities = [1, 2, 3, 4, 5, 6, 7, 8]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for q in qualities:
    if not os.path.exists(os.path.join(orig_path, f"{q}")):
        os.makedirs(os.path.join(orig_path, f"{q}"))
    if args.pretrained:
        model = msh(quality=q, pretrained=True)
    else:
        if args.fine_tuned:
            model = msh(quality=q, pretrained=True)
        else:
            model = MeanScaleHyperpriorGrayscale(*cfgs["mbt2018-mean"][q])
    
        model.load_state_dict(
            torch.load(
                os.path.join(
                    "..",
                    "models",
                    "training_models" if not args.fine_tuned else "finetuned_models", 
                    f"ms-hyperprior-{q}.pt"
                )
            )
        )

    model.update()
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for img, path in tqdm(dataloader):
            #if the directory does not exist create it
            paths = []
            for j in range(len(path[0])):
                for i in range(len(path)):
                    paths.append(path[i][j].replace("original", f"{q}_msh" if args.pretrained else f"{q}"))
            path = paths
            img = img.reshape(-1, 1, *shape).cuda()
            if args.pretrained or args.fine_tuned:
                img = img.repeat(1, 3, 1, 1)
            comp = model.compress(img)
            recon = model.decompress(comp["strings"], comp["shape"])
            recon = recon["x_hat"].detach().cpu().mean(1, keepdims=True)
            
            for sy, sz, p in zip(comp["strings"][0], comp["strings"][1], path):
                Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)
                with open(p[:-4]+".bin", "wb") as f:
                    pickle.dump([sy, sz, comp["shape"]], f)

            for r, p in zip(recon, path):
                with open(p, "wb") as f:
                    save_image(r, f)
