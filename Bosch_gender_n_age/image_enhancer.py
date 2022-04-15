import cv2
import numpy as np

def clahe(img):
    # bgr = cv2.imread(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = np.array(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2 ,tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def brightness_level(image, dim=50, thresh1=0.5, thresh2=0.4):
    # Resize image to 10x1
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = np.mean(L/np.max(L))
    # Return True if mean is greater than thresh else False
    # print(L)
    if L > thresh1:
        return 'bright'
    elif L > thresh2:
        return 'balanced'
    else:
        return 'dark'
