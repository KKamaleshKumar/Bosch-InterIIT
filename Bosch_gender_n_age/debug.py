import argparse
from distutils.dir_util import copy_tree
import os
from retinaface import RetinaFace
from retinaface.commons import postprocess
import numpy as np
import cv2
from ESRGAN.esrgan import Args_esr, esrgan
from GFPGAN.Gfpgan import Args_gfp, gfpgan
from SwinIR.swinir import swinir_test
from Zero_DCE.lowlight_test import lowlight
from PIL import Image
import shutil
import imghdr
import time
from PIL import Image
from image_enhancer import *
import torch
from dla import dla34
import onnxruntime as ort
import pandas as pd

from torchvision import transforms
def age_regression(img):
    img = cv2.resize(img,(256,256))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #img = np.transpose(img, (2,0,1))

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5946, 0.4548, 0.3891], std=[0.2191, 0.1949, 0.1856])
        ])

    img = transform(img)
    img = img.to(device)
    model = dla34().to(device)
    model.load_state_dict(torch.load('best_val.pth', map_location=device))
    model.eval()
    return model(img.unsqueeze(0)).item()

file_path = "/home/neham/inter_iit/Bosch_gender_n_age/input/99_1_2_20170117195405372.jpg.chip.jpg"
img = cv2.imread(file_path)
      


print(age_regression(img))