import argparse
from distutils.dir_util import copy_tree
import os
import imageio
# from retinaface import RetinaFace
# from retinaface.commons import postprocess
import numpy as np
import cv2
from GFPGAN.Gfpgan import Args_gfp, gfpgan
# from SwinIR.swinir import swinir_test
from Zero_DCE.lowlight_test import lowlight
from PIL import Image
import shutil
# import imghdr
import time
from PIL import Image
from image_enhancer import *
import torch
from dla import dla34
import onnxruntime as ort
import pandas as pd
# from GPEN.gpen import gpen
from Dsfd.test import dsfd
from tqdm import tqdm
from torchvision import transforms
# from basicsr.utils import imwrite
from torch.utils.data import Dataset, DataLoader, TensorDataset
import glob
from PIL import Image
from bbox_on_video import video_generator
from bbox_on_images import annotate_images


def init_age_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = dla34().to(device)
    model.load_state_dict(torch.load('age_regression_best_val.pth', map_location=device))
    return model    

def init_gender_model():
    ort_sess = ort.InferenceSession('efficientnet_lite.onnx')
    return ort_sess

def age_regression(img, model):
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img = cv2.resize(img,(256,256))
        # img = np.transpose(img, (2,0,1))

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5946, 0.4548, 0.3891], std=[0.2191, 0.1949, 0.1856])
            ])

        img = transform(img)
        img = img.to(device)
        model.eval()
    return model(img.unsqueeze(0)).item()


def gender_detect(img,ort_sess):
        img = cv2.resize(img, (384,384))
        img = img.astype(np.float16)
        img = (img-128)/128
        ort_inputs = {ort_sess.get_inputs()[0].name: np.array(img).astype(np.float32)[np.newaxis, ...]}
        outputs = ort_sess.run(None, ort_inputs)
        return outputs[0][0].argmax(0)


def handler(func, path, exc_info):
    pass


def empty_directory(dir_path):
    shutil.rmtree(dir_path, onerror=handler)
    os.makedirs(dir_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_folder', type=str,
                        default= './input', help='Input image folder path')
    parser.add_argument('-o', '--output', type=str,
                        default='./', help='Output folder')
    parser.add_argument('-v', '--video_folder', type=str,
                        default= './input', help = 'Input video folder path with one video')

    args = parser.parse_args()

    if os.path.exists('./gfpgan.egg-info') and not os.path.exists('./GFPGAN/gfpgan.egg-info'):
        os.makedirs('./GFPGsAN/gfpgan.egg-info')
        shutil.move('./gfpgan.egg-info', './GFPGAN/gfpgan.egg-info')

    if os.path.exists('./gfpgan.egg-info'):
        shutil.rmtree('./gfpgan.egg-info', onerror=handler)

    if os.path.exists('./realesrgan.egg-info') and not os.path.exists('./ESRGAN/realesrgan.egg-info'):
        os.makedirs('./ESRGAN/realesrgan.egg-info')
        shutil.move('./realesrgan.egg-info', './ESRGAN/realesrgan.egg-info')

    if os.path.exists('./realesrgan.egg-info'):
        shutil.rmtree('./realesrgan.egg-info', onerror=handler)

    empty_directory('./Zero_DCE/data/test_dataset')
    empty_directory('./Zero_DCE/data/resultset')
    empty_directory('./Dsfd/images')
    empty_directory('./cropped_faces')
    empty_directory('./results')
    empty_directory('./face_bbox')

    t0 = time.time()
    #copy_tree(args.input, './Zero_DCE/data/test_dataset/')
    check_ext_img = False
    check_ext_video = False
    if len(os.listdir('./input')) != 0:
        file = [x for x in os.listdir('./input')][0]
        check_ext_img = [True for x in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'] if file.rfind(x)>0]
        check_ext_video  = [True for x in [ 'mp4', 'mkv', 'gif'] if  file.rfind(x)> 0]
        if len(check_ext_video) > 0:
            check_ext_video = check_ext_video[0]
        if len(check_ext_img) > 0:
            check_ext_img = check_ext_img[0]

    if args.image_folder != './input' or check_ext_img:
        fr_num = 0
        original_images = {}
        for path in glob.glob(args.image_folder + '/*'): 
            original_images[f'frame{fr_num}.jpg'] = path.split('/')[-1]
            frame = cv2.imread(path)
            cv2.imwrite(f'./Zero_DCE/data/test_dataset/frame{fr_num}.jpg', frame)
            fr_num+=1 

    if args.video_folder != './input'  or check_ext_video:
        fr_num = 0
        video_path = [os.path.join(args.video_folder, x) for x in  os.listdir(args.video_folder)][0]
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f'./Zero_DCE/data/test_dataset/frame{fr_num}.jpg', frame)
                fr_num += 1
            else:
                break
        print(f'Total frames: {fr_num}')


    t1 = time.time()
    print('Writing files time: ', t1-t0)
   
    #LLE
    images = [os.path.join('./Zero_DCE/data/test_dataset', x) for x in os.listdir('./Zero_DCE/data/test_dataset')]
    for img_path in images:
    #   print(img_path)
      img = cv2.imread(img_path)
      brightness = brightness_level(img)
      if brightness == 'bright':
        img = clahe(img)
        cv2.imwrite(img_path.replace('test_dataset', 'resultset') ,img)
      elif brightness == 'dark':
        lowlight(img_path)
      else:
        shutil.copy(img_path, img_path.replace('test_dataset', 'resultset'))

    t2 = time.time()
    print("Image Enhancer: ", t2-t1)
    

    t3 = time.time()
    os.chdir('./Dsfd')
    ag = dsfd('../Zero_DCE/data/resultset')
    os.chdir('..')

    t4 = time.time()
    print('DSFD : ', t4 - t3)

    #GFPGAN
    torch.cuda.empty_cache()
    os.makedirs(args.output, exist_ok=True)
    os.chdir('./GFPGAN')
    args_gfp = Args_gfp(input='../cropped_faces',
                        output='../results/', suffix='out')
    gfpgan(args_gfp)
    os.chdir('..')
    t5 = time.time()
    print("GFPGAN: ", t5-t4)


    age_model = init_age_model()
    ort_sess = init_gender_model()

    try :
        images = [os.path.join('./results/restored_imgs', x) for x in os.listdir('./results/restored_imgs')]
    except :
        print('No face detected.')
        return
        
    os.makedirs(args.output, exist_ok=True)
    for i, img_path in enumerate(tqdm(images)):
      img = cv2.imread(img_path)
      img_path = img_path.split('/')[-1]
      fr_face = f'{int(img_path.split("_")[0][5:].replace(".jpg", ""))}_{int(img_path.split("_")[-1].replace(".jpg", ""))}'
      idx = ag.index[ag['fr_face'] == fr_face].values
      if len(idx) == 0:
          continue
      gender = 'F' if gender_detect(img, ort_sess) == 0 else 'M'
      img = cv2.resize(img,(256,256))
      age = age_regression(img, age_model) 
      idx = idx[0]
      ag.iloc[idx: idx+1 , 7: 8] = age
      ag.iloc[idx: idx+1 , 8: 9] = gender

    ag = ag.drop(columns= ['fr_face'], axis = 1)
    ag = ag.sort_values(by=['frame_num'], ascending=True)
    ag.reset_index(drop = True, inplace=True)
    if args.image_folder != './input' or check_ext_img:
        ag.to_csv(os.path.join(args.output,"results.csv"), index = False)
    else:
        ag.to_csv(os.path.join(args.output,video_path.split('/')[-1][:-3] + 'csv'), index = False)
    t6 = time.time()
    print(f'Finished csv results in {args.output}')   
    print('Age and gender: ', t6-t5)  
    print('Total time: ', t6-t0)  
    if args.video_folder != './input' or check_ext_video:
        video_generator(video_path = video_path, csv_path= os.path.join(args.output,video_path.split('/')[-1][:-3] + 'csv'), save_dir = args.output)
        print(f'Finished video results in {args.output}') 
    if args.image_folder != './input' or check_ext_img:
        annotate_images(image_folder_path = './Zero_DCE/data/resultset', csv_path = os.path.join(args.output,"results.csv"))
        print('Annotated images saved to face_bbox in root directory.') 

    empty_directory('./input')

if __name__ == '__main__':
    main()
