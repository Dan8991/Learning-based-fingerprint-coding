import pickle
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import PchipInterpolator
import numpy as np

def bj_delta(R1, PSNR1, R2, PSNR2, mode=0):
    if R1[0] > R1[-1]:
        R1 = R1[::-1]
        PSNR1 = PSNR1[::-1]
    if R2[0] > R2[-1]:
        R2 = R2[::-1]
        PSNR2 = PSNR2[::-1]
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # find integral
    if mode == 0:
        # spline fit
        spline1 = PchipInterpolator(lR1, PSNR1)
        spline2 = PchipInterpolator(lR2, PSNR2)

        # integration interval
        min_int = max(min(lR1), min(lR2))
        max_int = min(max(lR1), max(lR2))

        # indefinite integral of both polynomial curves
        int1 = spline1.integrate(min_int, max_int)
        int2 = spline2.integrate(min_int, max_int)

        # find avg diff between the areas to obtain the final measure
        avg_diff = (int2 - int1) / (max_int - min_int)
    else:
        # rate method: sames as previous one but with inverse order
        spline1 = PchipInterpolator(PSNR1, lR1)
        spline2 = PchipInterpolator(PSNR2, lR2)

        # integration interval
        min_int = max(min(PSNR1), min(PSNR2))
        max_int = min(max(PSNR1), max(PSNR2))

        # indefinite interval of both polynomial curves
        int1 = spline1.integrate(min_int, max_int)
        int2 = spline2.integrate(min_int, max_int)

        # find avg diff between the areas to obtain the final measure
        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        avg_diff = (np.exp(avg_exp_diff) - 1) * 100

    return avg_diff


with open('../results/final_dict.pickle', 'rb') as f:
    results = pickle.load(f)

for key in results["jpeg"].keys():
    results["jpeg"][key] = results["jpeg"][key][1:]

for key in results["msh"].keys():
    results["msh"][key] = results["msh"][key][:-1]

for key in results["msh_pretrained"].keys():
    results["msh_pretrained"][key] = results["msh_pretrained"][key][:-1]

matplotlib.rcParams.update({'font.size': 13})


results["jpeg"]["final_ssims"] = [x.cpu().item() for x in results["jpeg"]["final_ssims"]]
results["msh"]["final_ssims"] = [x.cpu().item() for x in results["msh"]["final_ssims"]]
results["msh_pretrained"]["final_ssims"] = [x.cpu().item() for x in results["msh_pretrained"]["final_ssims"]]
rate_jpeg = results["jpeg"]["final_rates"]
rate_msh = results["msh"]["final_rates"]
rate_mshp = results["msh_pretrained"]["final_rates"]
psnr_jpeg = results["jpeg"]["final_psnrs"]
psnr_msh = results["msh"]["final_psnrs"]
psnr_mshp = results["msh_pretrained"]["final_psnrs"]
ssim_jpeg = results["jpeg"]["final_ssims"]
ssim_msh = results["msh"]["final_ssims"]
ssim_mshp = results["msh_pretrained"]["final_ssims"]


plt.plot(results["jpeg"]["final_rates"], results["jpeg"]["final_ssims"], label="JPEG2000", c="green", marker="o", alpha=0.5)
plt.plot(results["msh"]["final_rates"], results["msh"]["final_ssims"], label="Finger-MSH", c="blue", marker="o", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_rates"], results["msh_pretrained"]["final_ssims"], label="MSH", c="red", marker="o", alpha=0.5)
plt.xlabel("Rate (bpp)")
plt.ylabel("SSIM")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/ssim.pdf", bbox_inches='tight')

plt.clf()

plt.plot(results["jpeg"]["final_rates"], results["jpeg"]["final_psnrs"], label="JPEG2000", c="green", marker="o", alpha=0.5)
plt.plot(results["msh"]["final_rates"], results["msh"]["final_psnrs"], label="Finger-MSH", c="blue", marker="o", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_rates"], results["msh_pretrained"]["final_psnrs"], label="MSH", c="red", marker="o", alpha=0.5)
plt.xlabel("Rate (bpp)")
plt.ylabel("PSNR (dB)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/psnr.pdf", bbox_inches='tight')

plt.clf()

results["jpeg"]["kept_term"] = np.array([results["jpeg"]["kept_term"][i] * 50 for i in range(len(results["jpeg"]["kept_term"]))])
results["msh"]["kept_term"] = np.array([results["msh"]["kept_term"][i] * 50 for i in range(len(results["msh"]["kept_term"]))])
results["msh_pretrained"]["kept_term"] = np.array([results["msh_pretrained"]["kept_term"][i] * 50 for i in range(len(results["msh_pretrained"]["kept_term"]))])
results["jpeg"]["kept_bif"] = np.array([results["jpeg"]["kept_bif"][i] * 50 for i in range(len(results["jpeg"]["kept_bif"]))])
results["msh"]["kept_bif"] = np.array([results["msh"]["kept_bif"][i] * 50 for i in range(len(results["msh"]["kept_bif"]))])
results["msh_pretrained"]["kept_bif"] = np.array([results["msh_pretrained"]["kept_bif"][i] * 50 for i in range(len(results["msh_pretrained"]["kept_bif"]))])


print(results["msh"]["kept_bif"], results["jpeg"]["kept_bif"], results["msh_pretrained"]["kept_bif"])
plt.plot(results["jpeg"]["final_rates"], results["jpeg"]["kept_bif"] + results["jpeg"]["kept_term"], label="JPEG2000", c="green", marker="*", alpha=0.5)
plt.plot(results["msh"]["final_rates"], results["msh"]["kept_bif"] + results["msh"]["kept_term"], label="Finger-MSH", c="blue", marker="x", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_rates"], results["msh_pretrained"]["kept_bif"] + results["msh_pretrained"]["kept_term"], label="MSH", c="red", marker="v", alpha=0.5)
plt.xlabel("Rate (bpp)")
plt.ylabel("Kept Minutiae (%)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/kept_rate.pdf", bbox_inches='tight')

plt.clf()

results["jpeg"]["extra_term"] = np.array([results["jpeg"]["extra_term"][i] * 50 for i in range(len(results["jpeg"]["extra_term"]))])
results["msh"]["extra_term"] = np.array([results["msh"]["extra_term"][i] * 50 for i in range(len(results["msh"]["extra_term"]))])
results["msh_pretrained"]["extra_term"] = np.array([results["msh_pretrained"]["extra_term"][i] * 50 for i in range(len(results["msh_pretrained"]["extra_term"]))])
results["jpeg"]["extra_bif"] = np.array([results["jpeg"]["extra_bif"][i] * 50 for i in range(len(results["jpeg"]["extra_bif"]))])
results["msh"]["extra_bif"] = np.array([results["msh"]["extra_bif"][i] * 50 for i in range(len(results["msh"]["extra_bif"]))])
results["msh_pretrained"]["extra_bif"] = np.array([results["msh_pretrained"]["extra_bif"][i] * 50 for i in range(len(results["msh_pretrained"]["extra_bif"]))])

plt.plot(results["jpeg"]["final_rates"], results["jpeg"]["extra_bif"] + results["jpeg"]["extra_term"], label="JPEG2000", c="green", marker="*", alpha=0.5)
plt.plot(results["msh"]["final_rates"], results["msh"]["extra_bif"] + results["msh"]["extra_term"], label="Finger-MSH", c="blue", marker="x", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_rates"], results["msh_pretrained"]["extra_bif"] + results["msh_pretrained"]["extra_term"], label="MSH", c="red", marker="v", alpha=0.5)
plt.xlabel("Rate (bpp)")
plt.ylabel("Extra Minutiae (%)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/extra_rate.pdf", bbox_inches='tight')

plt.clf()

results["jpeg"]["changed_to_bif"] = np.array([results["jpeg"]["changed_to_bif"][i] * 50 for i in range(len(results["jpeg"]["changed_to_bif"]))])
results["msh"]["changed_to_bif"] = np.array([results["msh"]["changed_to_bif"][i] * 50 for i in range(len(results["msh"]["changed_to_bif"]))])
results["msh_pretrained"]["changed_to_bif"] = np.array([results["msh_pretrained"]["changed_to_bif"][i] * 50 for i in range(len(results["msh_pretrained"]["changed_to_bif"]))])
results["jpeg"]["changed_to_term"] = np.array([results["jpeg"]["changed_to_term"][i] * 50 for i in range(len(results["jpeg"]["changed_to_term"]))])
results["msh"]["changed_to_term"] = np.array([results["msh"]["changed_to_term"][i] * 50 for i in range(len(results["msh"]["changed_to_term"]))])
results["msh_pretrained"]["changed_to_term"] = np.array([results["msh_pretrained"]["changed_to_term"][i] * 50 for i in range(len(results["msh_pretrained"]["changed_to_term"]))])

plt.plot(results["jpeg"]["final_rates"], results["jpeg"]["changed_to_bif"] + results["jpeg"]["changed_to_term"], label="JPEG2000", c="green", marker="*", alpha=0.5)
plt.plot(results["msh"]["final_rates"], results["msh"]["changed_to_bif"] + results["msh"]["changed_to_term"], label="Finger-MSH", c="blue", marker="x", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_rates"], results["msh_pretrained"]["changed_to_bif"] + results["msh"]["changed_to_term"], label="MSH", c="red", marker="v", alpha=0.5)
# plt.plot(results["jpeg"]["final_rates"], results["jpeg"]["changed_to_term"], label="JPEG2000(Bif2Term)", c="green", marker=".", alpha=0.5, linestyle="--")
# plt.plot(results["msh"]["final_rates"], results["msh"]["changed_to_term"], label="Finger-MSH(Bif2Term)", c="blue", marker=".", alpha=0.5, linestyle="--")
# plt.plot(results["msh_pretrained"]["final_rates"], results["msh_pretrained"]["changed_to_term"], label="MSH(Bif2Term)", c="red", marker=".", alpha=0.5, linestyle="--")
plt.xlabel("Rate (bpp)")
plt.ylabel("Changed (%)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/changed_rate.pdf", bbox_inches='tight')

plt.clf()

plt.plot(results["jpeg"]["final_psnrs"], results["jpeg"]["kept_bif"] +  results["jpeg"]["kept_term"], label="JPEG2000", c="green", marker="*", alpha=0.5)
plt.plot(results["msh"]["final_psnrs"], results["msh"]["kept_bif"] +  results["msh"]["kept_term"], label="Finger-MSH", c="blue", marker="x", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_psnrs"], results["msh_pretrained"]["kept_bif"] +  results["msh_pretrained"]["kept_term"], label="MSH", c="red", marker="v", alpha=0.5)
plt.xlabel("PSNR (dB)")
plt.ylabel("Kept Minutiae (%)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/kept_psnr.pdf", bbox_inches='tight')

plt.clf()

plt.plot(results["jpeg"]["final_psnrs"], results["jpeg"]["extra_bif"] + results["jpeg"]["extra_term"], label="JPEG2000", c="green", marker="*", alpha=0.5)
plt.plot(results["msh"]["final_psnrs"], results["msh"]["extra_bif"] + results["msh"]["extra_term"], label="Finger-MSH", c="blue", marker="x", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_psnrs"], results["msh_pretrained"]["extra_bif"] + results["msh_pretrained"]["extra_term"], label="MSH", c="red", marker="v", alpha=0.5)
plt.xlabel("PSNR (dB)")
plt.ylabel("Extra Minutiae (%)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/extra_psnr.pdf", bbox_inches='tight')

plt.clf()

plt.plot(results["jpeg"]["final_psnrs"], results["jpeg"]["changed_to_bif"] + results["jpeg"]["changed_to_term"], label="JPEG2000", c="green", marker="*", alpha=0.5)
plt.plot(results["msh"]["final_psnrs"], results["msh"]["changed_to_bif"] + results["msh"]["changed_to_term"], label="Finger-MSH", c="blue", marker="x", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_psnrs"], results["msh_pretrained"]["changed_to_bif"] + results["msh_pretrained"]["changed_to_term"], label="MSH", c="red", marker="v", alpha=0.5)
plt.xlabel("PSNR (dB)")
plt.ylabel("Changed (%)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/changed_psnr.pdf", bbox_inches='tight')

plt.clf()

plt.plot(results["jpeg"]["final_ssims"], results["jpeg"]["kept_bif"] + results["jpeg"]["kept_term"], label="JPEG2000", c="green", marker="*", alpha=0.5)
plt.plot(results["msh"]["final_ssims"], results["msh"]["kept_bif"] + results["msh"]["kept_term"], label="Finger-MSH", c="blue", marker="x", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_ssims"], results["msh_pretrained"]["kept_bif"] + results["msh_pretrained"]["kept_term"], label="MSH", c="red", marker="v", alpha=0.5)
plt.xlabel("SSIM")
plt.ylabel("Kept Minutiae (%)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/kept_ssim.pdf", bbox_inches='tight')

plt.clf()

plt.plot(results["jpeg"]["final_ssims"], results["jpeg"]["extra_bif"] + results["jpeg"]["extra_term"], label="JPEG2000", c="green", marker="*", alpha=0.5)
plt.plot(results["msh"]["final_ssims"], results["msh"]["extra_bif"] + results["msh"]["extra_term"], label="Finger-MSH", c="blue", marker="x", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_ssims"], results["msh_pretrained"]["extra_bif"] + results["msh_pretrained"]["extra_term"], label="MSH", c="red", marker="v", alpha=0.5)
plt.xlabel("SSIM")
plt.ylabel("Extra Minutiae (%)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/extra_ssim.pdf", bbox_inches='tight')

plt.clf()

plt.plot(results["jpeg"]["final_ssims"], results["jpeg"]["changed_to_bif"] + results["jpeg"]["changed_to_term"], label="JPEG2000", c="green", marker="*", alpha=0.5)
plt.plot(results["msh"]["final_ssims"], results["msh"]["changed_to_bif"] + results["msh"]["changed_to_term"], label="Finger-MSH", c="blue", marker="x", alpha=0.5)
plt.plot(results["msh_pretrained"]["final_ssims"], results["msh_pretrained"]["changed_to_bif"] + results["msh_pretrained"]["changed_to_term"], label="MSH", c="red", marker="v", alpha=0.5)
plt.xlabel("SSIM")
plt.ylabel("Changed (%)")
plt.tight_layout()
plt.legend()
plt.savefig("../figures/changed_ssim.pdf", bbox_inches='tight')