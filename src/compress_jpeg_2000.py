from torchvision import transforms
from dataset import Fingerprint
from torchvision.transforms.functional import to_pil_image
import torch
from tqdm import tqdm
from pathlib import Path
import os
import argparse
from torchvision.utils import save_image



parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/mnt/Dataset/original")
args = parser.parse_args()

orig_path = args.path
shape = (320, 320)

dataset = Fingerprint(orig_path, transform=transforms.Compose([transforms.CenterCrop(shape), transforms.ToTensor()]))
jpeg_qualities = [2, 5, 10, 20, 30, 40]

for q in jpeg_qualities:
    print(f"current q: {q}")

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            img, path = dataset[i]
            #if the directory does not exist create it
            compressed_paths = []
            for i in range(len(path)):
                compressed_paths.append(path[i].replace("original", f"{q}_jpeg"))
                Path(os.path.dirname(compressed_paths[i])).mkdir(parents=True, exist_ok=True)
            
            img = img.reshape(-1, 1, *shape)

            for i, pc in zip(img, compressed_paths):
                Path(os.path.dirname(pc.replace(f"{q}_jpeg", f"cropped"))).mkdir(parents=True, exist_ok=True)
                save_image(i, pc.replace("jp2", "tif").replace(f"{q}_jpeg", f"cropped"))
                i = to_pil_image(i).convert("L")
                i.save(pc[:-4]+".jp2", "JPEG2000",   quality_mode='rates', quality_layers=[q])

