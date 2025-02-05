import numpy as np
import os
import random
from tqdm import tqdm

# Define paths
data_dir = "ProcessedData0"
lungs_dir = os.path.join(data_dir, "lungs_roi/")
masks_dir = os.path.join(data_dir, "nodule_mask/")

# Load and shuffle filenames
lung_names = os.listdir(lungs_dir)

# Split data into training and test sets (80% train, 20% test)
n = len(lung_names)
s = int(n * 0.8)
train_lung_names = lung_names[:s]
test_lung_names = lung_names[s:]

print(test_lung_names)

print("Number of training samples:", len(train_lung_names))
print("Number of test samples:", len(test_lung_names))

# Initialize lists for training and test data
trainX, trainY = [], []
testX, testY = [], []

# Load training data
for lname in tqdm(train_lung_names, desc="Loading training data"):
    mname = lname.replace("lungs", "masks")
    lung = np.load(os.path.join(lungs_dir, lname))
    mask = np.load(os.path.join(masks_dir, mname))
    trainX.append(lung)
    trainY.append(mask)

# Convert lists to arrays and save
trainX = np.array(trainX, dtype=np.uint8)
trainY = np.array(trainY, dtype=np.uint8)
np.save("trainX.npy", trainX)
np.save("trainY.npy", trainY)

# Load test data
for lname in tqdm(test_lung_names, desc="Loading test data"):
    mname = lname.replace("lungs", "masks")
    lung = np.load(os.path.join(lungs_dir, lname))
    mask = np.load(os.path.join(masks_dir, mname))
    testX.append(lung)
    testY.append(mask)

# Convert lists to arrays and save
testX = np.array(testX, dtype=np.uint8)
testY = np.array(testY, dtype=np.uint8)
np.save("testX.npy", testX)
np.save("testY.npy", testY)

print(f"trainX: {len(trainX)}")
print(f"trainY: {len(trainY)}")
print(f"testX: {len(testX)}")
print(f"testY: {len(testY)}")

print("Data preparation complete. Arrays saved as trainX.npy, trainY.npy, testX.npy, testY.npy.")
