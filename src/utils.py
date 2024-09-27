import fingerprint_feature_extractor
import os
import fingerprint_enhancer
from multiprocessing import Pool
import cv2
import numpy as np
import skimage
from scipy.spatial.distance import cdist

def saveResult(self, FeaturesTerm, FeaturesBif, path):
    (rows, cols) = self._skel.shape
    DispImg = np.zeros((rows, cols, 3), np.uint8)
    DispImg[:, :, 0] = 255 * self._skel
    DispImg[:, :, 1] = 255 * self._skel
    DispImg[:, :, 2] = 255 * self._skel

    for idx, curr_minutiae in enumerate(FeaturesTerm):
        row, col = curr_minutiae.locX, curr_minutiae.locY
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
        skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

    for idx, curr_minutiae in enumerate(FeaturesBif):
        row, col = curr_minutiae.locX, curr_minutiae.locY
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
        skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))
    cv2.imwrite(path, DispImg)

def extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, save=False, path=None):
    feature_extractor = fingerprint_feature_extractor.FingerprintFeatureExtractor()
    feature_extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
    if (invertImage):
        img = 255 - img
    FeaturesTerm, FeaturesBif = feature_extractor.extractMinutiaeFeatures(img)

    if (save):
        saveResult(feature_extractor, FeaturesTerm, FeaturesBif, path)

    if(showResult):
        feature_extractor.showResults(FeaturesTerm, FeaturesBif)

    return [FeaturesTerm, FeaturesBif]

def enhance_fingerprints(imgs):
    imgs = imgs.numpy()
    shape = imgs.shape
    imgs = imgs.reshape(-1, imgs.shape[-2], imgs.shape[-1])
    enhanced_imgs = []
    with Pool(os.cpu_count()) as p:
        enhanced_imgs = p.map(fingerprint_enhancer.enhance_Fingerprint, imgs)
    enhanced_images = np.stack(enhanced_imgs).reshape(shape)
    return enhanced_images

def compute_number_of_minutiaes(t1, b1, t2, b2, threshold=3):
    feat_term1 = [(m.locX, m.locY) for m in t1] 
    feat_bif1 = [(m.locX, m.locY) for m in b1]
    feat_term2 = [(m.locX, m.locY) for m in t2]
    feat_bif2 = [(m.locX, m.locY) for m in b2]
    
    # Convert to numpy arrays
    feat_term1 = np.array(feat_term1)
    feat_bif1 = np.array(feat_bif1)
    feat_term2 = np.array(feat_term2)
    feat_bif2 = np.array(feat_bif2)

        # Calculate pairwise distances between minutiae points
    if feat_term1.shape[0] == 0 or feat_term2.shape[0] == 0:
        distances_term = np.ones((1,1)) * np.inf
    else:
        distances_term = cdist(feat_term1, feat_term2)
    if feat_bif1.shape[0] == 0 or feat_bif2.shape[0] == 0:
        distances_bif = np.ones((1,1)) * np.inf
    else:
        distances_bif = cdist(feat_bif1, feat_bif2)
    if feat_term1.shape[0] == 0 or feat_bif2.shape[0] == 0:
        distances_term_bif = np.ones((1,1)) * np.inf
    else:
        distances_term_bif = cdist(feat_term1, feat_bif2)
    if feat_bif1.shape[0] == 0 or feat_term2.shape[0] == 0:
        distances_bif_term = np.ones((1,1)) * np.inf
    else:
        distances_bif_term = cdist(feat_bif1, feat_term2)
    
    # Find matches based on distance threshold
    matches_term = distances_term < threshold
    matches_bif = distances_bif < threshold
    matches_term_bif = distances_term_bif < threshold
    matches_bif_term = distances_bif_term < threshold

    return (np.sum(matches_term, 1) > 0).sum(), (np.sum(matches_bif, 1) > 0).sum(), (np.sum(matches_bif_term, 1) > 0).sum(), (np.sum(matches_term_bif, 1) > 0).sum()