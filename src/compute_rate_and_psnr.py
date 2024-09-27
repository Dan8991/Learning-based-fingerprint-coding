from torchvision import transforms
from dataset import Fingerprint
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
from utils import enhance_fingerprints, compute_number_of_minutiaes, extract_minutiae_features
from multiprocessing import Pool
from torchmetrics.image import StructuralSimilarityIndexMeasure
import pickle
import argparse

def compute_metrics(model_string, final_dict, extra_str, qualities, extension="bmp", args=None):
    ranges = []

    kept_bif = []
    kept_term = []
    extra_term = []
    extra_bif = []
    missing_term = []
    missing_bif = []
    changed_to_bif = []
    changed_to_term = []
    final_ssims = []
    final_rates = []
    final_psnrs = []

    for q in qualities:
        new_path = args.path.replace("original", f"{q}{extra_str}")
        dataset_compressed = Fingerprint(
            new_path,
            transform=transforms.Compose([transforms.CenterCrop(shape), transforms.ToTensor()]),
            extension=extension
        )

        print("Dataset size:", len(dataset_compressed))
        
        compressed_dataloader = DataLoader(dataset_compressed, batch_size=8, shuffle=False, num_workers=2)
        
        psnrs = []
        rates = []
        ssims = []
        kept_bif.append([])
        kept_term.append([])
        extra_term.append([])
        extra_bif.append([])
        missing_term.append([])
        missing_bif.append([])
        changed_to_bif.append([])
        changed_to_term.append([])

        for i, (d, dc) in tqdm(enumerate(zip(dataloader, compressed_dataloader))):
            d, path_d = d
            dc, path_dc = dc
            path_d = [p for ps in path_d for p in ps]
            path_dc = [p for ps in path_dc for p in ps]
            d = d.to(device)
            dc = dc.to(device)
            dc = dc.mean(dim = 4, keepdims = True)
            d = d.reshape(-1, 1, *shape)
            dc = dc.reshape(-1, 1, *shape)
            ssims.append(ssim(d, dc))
            de = enhance_fingerprints(d.squeeze(1).cpu().detach())
            dec = enhance_fingerprints(dc.squeeze(1).cpu().detach())
            with Pool(os.cpu_count()) as p:
                f1 = p.map(extract_minutiae_features, de)
                f2 = p.map(extract_minutiae_features, dec)
                matches = p.starmap(compute_number_of_minutiaes, [[t1, b1, t2, b2] for (t1, b1), (t2, b2) in zip(f1, f2)])
            for i in range(len(f1)):
                t1, b1 = f1[i]
                t2, b2 = f2[i]
                b1 = max(1, len(b1))
                b2 = max(1, len(b2))
                t1 = max(1, len(t1))
                t2 = max(1, len(t2))
                kt, kb, ct, cb = matches[i]
                kept_bif[-1].append(kb/b1)
                kept_term[-1].append(kt/t1)
                extra_term[-1].append((t2 - kt) / t2)
                extra_bif[-1].append((b2 - kb) / b2)
                missing_term[-1].append((t1 - kt) / t1)
                missing_bif[-1].append((b1 - kb) / b1)
                changed_to_bif[-1].append(cb/t2)
                changed_to_term[-1].append(ct/b2)
            d = d.reshape(d.shape[0], -1)
            dc = dc.reshape(dc.shape[0], -1)

            #compute psnr between each pair of images
            psnr = 10 * torch.log10(1 / torch.mean((d - dc)**2, dim = 1))
            psnrs.extend(list(psnr.cpu()))
            rate_ext = ".bin" if extension != "jp2" else ".jp2"
            rates.extend([os.path.getsize(p[:-4]+rate_ext)*8 / num_pixels for p in path_dc])
        final_psnrs.append(torch.mean(torch.tensor(psnrs)))
        psnrs = torch.tensor(psnrs)

        ranges.append([torch.mean(psnrs) - torch.min(psnrs), torch.max(psnrs) - torch.mean(psnrs)])
        final_rates.append(torch.mean(torch.tensor(rates)))
        final_ssims.append(torch.mean(torch.cat(ssims)))

    kept_bif = [torch.mean(torch.tensor(kb)) for kb in kept_bif]
    kept_term = [torch.mean(torch.tensor(kt)) for kt in kept_term]
    extra_term = [torch.mean(torch.tensor(et)) for et in extra_term]
    extra_bif = [torch.mean(torch.tensor(eb)) for eb in extra_bif]
    missing_term = [torch.mean(torch.tensor(mt)) for mt in missing_term]
    missing_bif = [torch.mean(torch.tensor(mb)) for mb in missing_bif]
    changed_to_bif = [torch.mean(torch.tensor(cb)) for cb in changed_to_bif]
    changed_to_term = [torch.mean(torch.tensor(ct)) for ct in changed_to_term]


    final_dict[model_string] = {}
    final_dict[model_string]["kept_bif"] = kept_bif
    final_dict[model_string]["kept_term"] = kept_term
    final_dict[model_string]["extra_term"] = extra_term
    final_dict[model_string]["extra_bif"] = extra_bif
    final_dict[model_string]["missing_term"] = missing_term
    final_dict[model_string]["missing_bif"] = missing_bif
    final_dict[model_string]["changed_to_bif"] = changed_to_bif
    final_dict[model_string]["changed_to_term"] = changed_to_term
    final_dict[model_string]["final_psnrs"] = final_psnrs
    final_dict[model_string]["final_rates"] = final_rates
    final_dict[model_string]["final_ssims"] = final_ssims
    
    return final_dict

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/mnt/Dataset/original")
parser.add_argument("--dataset", type=str, default="Casia")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#removing part of the sensor to avoid biasing distortion due to overfitting
shape = (320, 270)

dataset = Fingerprint(args.path, transform=transforms.Compose([transforms.CenterCrop(shape), transforms.ToTensor()]))

dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

num_pixels = 320*320

ssim = StructuralSimilarityIndexMeasure(reduction="none")
ssim = ssim.to(device)

fig, axs = plt.subplots(4, 4, figsize=(20, 10))

final_rates = []
final_psnrs = []
final_ssims = []

final_dict = {}

jpeg_qualities = [2, 5, 10, 20, 30, 40]
final_dict = compute_metrics("jpeg", final_dict, "_jpeg", jpeg_qualities, args=args, extension="jp2")

qualities = [1, 2, 3, 4, 5, 6, 7, 8]
final_dict = compute_metrics("msh_pretrained", final_dict, "_msh", qualities, args=args)
final_dict = compute_metrics("msh", final_dict, "", qualities, args=args)

with open(os.path.join("..", "results", f"final_dict.pickle"), "wb") as f:
    pickle.dump(final_dict, f)

