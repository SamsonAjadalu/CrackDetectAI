import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split  # Still used for splitting data

ANNOTATION_PATH = 'dataset/annotations'
IMAGE_PATH = 'dataset/images'
FIXED_SIZE = (64, 64)  # Resize all ROIs to this fixed size

def extract_rois_and_labels(annotation_path, image_path):
    rois = []
    labels = []

    for xml_file in os.listdir(annotation_path):
        if xml_file.endswith('.xml'):
            xml_file_path = os.path.join(annotation_path, xml_file)
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            image_file = os.path.join(image_path, xml_file.replace('.xml', '.jpg'))
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            for obj in root.findall('object'):
                label = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))

                roi = image[ymin:ymax, xmin:xmax]
                if roi.size == 0:
                    continue

                rois.append(roi)
                labels.append(label)

    return rois, labels

def augment_roi(roi):
    augmented_rois = [roi]
    augmented_rois.append(cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE))
    augmented_rois.append(cv2.rotate(roi, cv2.ROTATE_180))
    augmented_rois.append(cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE))
    augmented_rois.append(cv2.flip(roi, 0))
    augmented_rois.append(cv2.flip(roi, 1))
    return augmented_rois

# Extract and augment ROIs and labels
print("Starting ROI extraction...")
rois, labels = extract_rois_and_labels(ANNOTATION_PATH, IMAGE_PATH)
augmented_rois, augmented_labels = [], []

for roi, label in zip(rois, labels):
    for aug_roi in augment_roi(roi):
        augmented_rois.append(aug_roi)
        augmented_labels.append(label)

print(f"Extracted and augmented {len(augmented_rois)} ROIs.")

# Initialize HOG descriptor with a smaller window size for the resized images
hog = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

# Feature extraction with HOG
descriptor_list = []
final_labels = []

print("Extracting HOG features...")
for i, (roi, label) in enumerate(zip(augmented_rois, augmented_labels)):
    resized_roi = cv2.resize(roi, FIXED_SIZE)  # Resize ROI to FIXED_SIZE
    hog_descriptor = hog.compute(resized_roi)  # Compute HOG descriptor
    descriptor_list.append(hog_descriptor.flatten())  # Flatten to 1D array
    final_labels.append(label)
    print(f"Processed HOG for ROI {i + 1}/{len(augmented_rois)}")

# Convert lists to numpy arrays
descriptor_array = np.array(descriptor_list, dtype=np.float32)
final_labels = np.array(final_labels)

# Encode labels into integers (OpenCV requires numeric labels)
unique_labels = {label: idx for idx, label in enumerate(set(final_labels))}
encoded_labels = np.array([unique_labels[label] for label in final_labels], dtype=np.int32)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(descriptor_array, encoded_labels, test_size=0.3, random_state=42)

# Initialize OpenCV's k-NN model
knn = cv2.ml.KNearest_create()

# Train the k-NN model
print("Training k-NN classifier...")
knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# Evaluate model
correct = 0
total = len(X_test)

for i in range(total):
    sample = X_test[i].reshape(1, -1)
    _, results, _, _ = knn.findNearest(sample, k=1)
    predicted_label = int(results[0][0])
    if predicted_label == y_test[i]:
        correct += 1

accuracy = (correct / total) * 100
print(f"Model accuracy: {accuracy:.2f}%")
