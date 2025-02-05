# imports 
import os
import cv2
import copy
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import SimpleITK as stk
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from skimage import measure
import os


root = os.path.join(os.getcwd(), "DATA/")
target_root = os.path.join(os.getcwd(), "ProcessedData/")


subset = 7
file_list = glob(root+f"subset{subset}/*.mhd")
annotations_df = pd.read_csv(root+"annotations.csv")
print("Files Count:",len(file_list))
print("DF Count:",len(annotations_df))
annotations_df.head()

# Just to visulaize 

#d = annotations_df['diameter_mm'].values
#plt.hist(d, bins=80)

# To filter file name that are present in both csv and subset

def get_filename(file_list, file):
    for f in file_list:
        if file in f:
            return f
        
annotations_df["filename"] = annotations_df["seriesuid"].map(lambda file: get_filename(file_list, file))
annotations_df = annotations_df.dropna()
annotations_df = annotations_df[annotations_df['diameter_mm']>=3.9]     # Excluding nodules with diameter less than 3.9mm
print(len(annotations_df))
annotations_df.to_excel(target_root+"annotations_filtered.xlsx",index=True)

print()
def load_mhd(file):
    mhdimage = stk.ReadImage(file)
    ct_scan = stk.GetArrayFromImage(mhdimage)
    origin = np.array(list(mhdimage.GetOrigin()))
    space = np.array(list(mhdimage.GetSpacing()))
    return ct_scan, origin, space


def make_mask(img, center, diam):
    mask = np.zeros_like(img, dtype=np.uint8)
    mask = cv2.circle(mask, (abs(int(center[0])),abs(int(center[1]))), abs(diam//2), 255, -1)
    return mask

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # CLAHE(Contrast Limited Adaptive Histogram Equalization) filter for enhancing the contrast of an image

# Iterating over all the files in the subset
for i,file in tqdm(enumerate(np.unique(annotations_df['filename'].values))):
    annotations = annotations_df[annotations_df["filename"]==file]
    ct, origin, space = load_mhd(file)      # Loading the CT scan
    num_z, height, width = ct.shape
    ct_norm = cv2.normalize(ct, None, 0, 255, cv2.NORM_MINMAX)   # Normalizing the CT scan
    for idx, row in annotations.iterrows():
        node_x = int(row["coordX"])     # X coordinate of the nodule
        node_y = int(row["coordY"])     # Y coordinate of the nodule
        node_z = int(row["coordZ"])     # Z coordinate of the nodule
        diam = int(row["diameter_mm"])  # Diameter of the nodule

        center = np.array([node_x, node_y, node_z])   # nodule center (x,y,z)
        v_center = np.rint((center-origin)/space)   # nodule center in voxel space (still x,y,z ordering)

        img_norm = ct_norm[int(v_center[2]),:,:]    # a slice of the CT scan containing the nodule
        img_norm = cv2.resize(img_norm, (512,512))  # Resizing the CT scan to 512x512
        img_norm_improved = clahe.apply(img_norm.astype(np.uint8))  # Applying CLAHE filter to the image

        ################################################################################################
        v_diam = int(diam/space[0])+5       # Diameter of the nodule in voxel space
        mask = make_mask(img_norm, v_center, v_diam)    # Creating a mask of the nodule
        if v_diam>18:       # If the nodule is too big, we will also take neighboring slices
            img_norm2 = ct_norm[(int(v_center[2])-1),:,:]
            img_norm2 = cv2.resize(img_norm2, (512,512))
            img_norm2_improved = clahe.apply(img_norm2.astype(np.uint8))
            mask2 = make_mask(img_norm2, v_center, v_diam-1)
            
            img_norm3 = ct_norm[(int(v_center[2])+1),:,:]
            img_norm3 = cv2.resize(img_norm3, (512,512))
            img_norm3_improved = clahe.apply(img_norm3.astype(np.uint8))
            mask3 = make_mask(img_norm3, v_center, v_diam-1)
            
            
        # Calculating the threshold value for extracting the nodule mask using binary thresholding
        mask = cv2.bitwise_and(img_norm, img_norm, mask=cv2.dilate(mask,kernel=np.ones((5,5))))
        pts = mask[mask>0]
        kmeans2 = KMeans(n_clusters=2).fit(np.reshape(pts,(len(pts),1)))
        centroids2 = sorted(kmeans2.cluster_centers_.flatten())
        threshold2 = np.mean(centroids2)
        
        _, mask = cv2.threshold(mask, threshold2, 255, cv2.THRESH_BINARY)
        if v_diam>18:
            mask2 = cv2.bitwise_and(img_norm2, img_norm2, mask=cv2.dilate(mask2,kernel=np.ones((5,5))))
            _, mask2 = cv2.threshold(mask2, threshold2, 255, cv2.THRESH_BINARY)
            
            mask3 = cv2.bitwise_and(img_norm3, img_norm3, mask=cv2.dilate(mask3,kernel=np.ones((5,5))))
            _, mask3 = cv2.threshold(mask3, threshold2, 255, cv2.THRESH_BINARY)
        ################################################################################################
        
        # Calculating the threshold value to segment the lungs from CT scan slices using binary thresholding
        centeral_area = img_norm[100:400, 100:400]
        kmeans = KMeans(n_clusters=2).fit(np.reshape(centeral_area, [np.prod(centeral_area.shape), 1]))
        centroids = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centroids)
        
        # Steps to segment the lungs from CT scan slices
        ret, lung_roi = cv2.threshold(img_norm, threshold, 255, cv2.THRESH_BINARY_INV)
        lung_roi = cv2.erode(lung_roi, kernel=np.ones([4,4]))
        lung_roi = cv2.dilate(lung_roi, kernel=np.ones([13,13]))
        lung_roi = cv2.erode(lung_roi, kernel=np.ones([8,8]))

        labels = measure.label(lung_roi)        # Labelling different regions in the image
        regions = measure.regionprops(labels)   # Extracting the properties of the regions
        good_labels = []
        for prop in regions:        # Filtering the regions that are not too close to the edges
            B = prop.bbox           # Regions that are too close to the edges are outside regions of lungs
            if B[2]-B[0] < 475 and B[3]-B[1] < 475 and B[0] > 40 and B[2] < 472:
                good_labels.append(prop.label)
        lung_roi_mask = np.zeros_like(labels)
        for N in good_labels:
            lung_roi_mask = lung_roi_mask + np.where(labels == N, 1, 0)

        # Steps to get proper segmentation of the lungs without noise and holes
        contours, hirearchy = cv2.findContours(lung_roi_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        external_contours = np.zeros(lung_roi_mask.shape)
        for i in range(len(contours)):
            if hirearchy[0][i][3] == -1:  #External Contours
                area = cv2.contourArea(contours[i])
                if area>518.0:
                    cv2.drawContours(external_contours,contours,i,(1,1,1),-1)
        external_contours = cv2.dilate(external_contours, kernel=np.ones([4,4]))
        
        external_contours = cv2.bitwise_not(external_contours.astype(np.uint8))
        external_contours = cv2.erode(external_contours, kernel=np.ones((7,7)))
        external_contours = cv2.bitwise_not(external_contours)
        external_contours = cv2.dilate(external_contours, kernel=np.ones((12,12)))
        external_contours = cv2.erode(external_contours, kernel=np.ones((12,12)))
        
        img_norm_improved = img_norm_improved.astype(np.uint8)
        external_contours = external_contours.astype(np.uint8)      # Final segmentated lungs mask
        extracted_lungs = cv2.bitwise_and(img_norm_improved, img_norm_improved, mask=external_contours)
        
        #################################################################################################
        if v_diam>18:
            img_norm2_improved = img_norm2_improved.astype(np.uint8)
            extracted_lungs2 = cv2.bitwise_and(img_norm2_improved, img_norm2_improved, mask=external_contours)
            mask2 = mask2.astype(np.uint8)
            np.save(os.path.join(target_root+"nodule_mask/", f"masks_{subset}_{i}_{idx}_2.npy"), mask2)
            np.save(os.path.join(target_root+"lungs_roi/", f"lungs_{subset}_{i}_{idx}_2.npy"), extracted_lungs2)
            
            img_norm3_improved = img_norm3_improved.astype(np.uint8)
            extracted_lungs3 = cv2.bitwise_and(img_norm3_improved, img_norm3_improved, mask=external_contours)
            mask3 = mask3.astype(np.uint8)
            np.save(os.path.join(target_root+"nodule_mask/", f"masks_{subset}_{i}_{idx}_3.npy"), mask3)
            np.save(os.path.join(target_root+"lungs_roi/", f"lungs_{subset}_{i}_{idx}_3.npy"), extracted_lungs3)
        #################################################################################################
        
        mask = mask.astype(np.uint8)
        
        np.save(os.path.join(target_root+"nodule_mask/", f"masks_{subset}_{i}_{idx}.npy"), mask)
        np.save(os.path.join(target_root+"lungs_roi/", f"lungs_{subset}_{i}_{idx}.npy"), extracted_lungs)