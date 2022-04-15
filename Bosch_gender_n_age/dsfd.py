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

class Dataset(Dataset):
    def __init__(self, path, ort):
        self.files = { os.path.join(path, x): x.replace('.jpg', '')  for x in sorted(os.listdir(path))}
        # print(self.files)
        self.ort = ort
        # self.labels = [filepath.split('/')[-2] for filepath in self.files]
    def __getitem__(self, item):
        item_key = list(self.files.keys())[item]
        item_value = list(self.files.values())[item]
        img = cv2.imread(item_key)
        
        img_age = cv2.resize(img,(256,256))
        # img = np.transpose(img, (2,0,1))

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5946, 0.4548, 0.3891], std=[0.2191, 0.1949, 0.1856])
            ])

        img_age = transform(img_age)

        img_gender = cv2.resize(img, (384,384))
        img_gender = (img_gender-128)/128
        gender_dict = {self.ort.get_inputs()[0].name: np.array(img_gender).astype(np.float32)[np.newaxis, ...]}
        # print(gender_dict['efficientnetv2_s_input:0'].shape)
        return [ img_age, gender_dict, item_value]
        # return img_age
    def __len__(self):
        return len(self.files)

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
        img = (img)/256
        ort_inputs = {ort_sess.get_inputs()[0].name: np.array(img).astype(np.float32)[np.newaxis, ...]}
        outputs = ort_sess.run(None, img)#,ort_inputs)
        return outputs[0][0].argmax(0)
        # return outputs


def handler(func, path, exc_info):
    pass


def empty_directory(dir_path):
    shutil.rmtree(dir_path, onerror=handler)
    os.makedirs(dir_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_folder', type=str,
                        default=None, help='Input image folder path')
    parser.add_argument('-o', '--output', type=str,
                        default='./', help='Output folder')
    parser.add_argument('-v', '--video_folder', type=str,
                        default= None, help = 'Input video folder path with one video')

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

    # empty_directory('./Zero_DCE/data/test_dataset')
    # empty_directory('./Zero_DCE/data/resultset')
    # empty_directory('./Dsfd/images')
    # empty_directory('./cropped_faces')
    # empty_directory('./results')

    t0 = time.time()
    # ort_sess = init_gender_model()
    # ag = pd.read_csv('./video1.csv')
    # dataset = Dataset(path = './results/restored_imgs', ort= ort_sess)
    # train_ldr = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    # output_dict = {}
    # age_model = init_age_model()
    # ort_sess = init_gender_model()
    # print(len(os.listdir('./results/restored_imgs')))
    # for batch, data in enumerate(tqdm(train_ldr)):
    #     #'efficientnetv2_s_input:0'
    #     print(batch, data[1]['efficientnetv2_s_input:0'].shape, data[0].shape)

    #     torch.cuda.empty_cache()
    #     age_input = data[0]
    #     gender_input = {ort_sess.get_inputs()[0].name : data[1]['efficientnetv2_s_input:0'].squeeze().detach().numpy()}     
    #     age_output = age_regression(age_input, age_model)
    #     gender_output = gender_detect(gender_input, ort_sess)
    #     del age_output
    #     print(gender_output[0].shape, len(gender_output), age_output.shape, len(data[2]))
    #     del gender_output
    #     # print(age_output.shape)
    #     break
    #     # output_dict[batch]
    # print('Dataset creation time:' , time.time() - t0)
    # return
    #copy_tree(args.input, './Zero_DCE/data/test_dataset/')
#     if args.image_folder is not None:
#         fr_num = 0
#         original_images = {}
#         for path in glob.glob(args.image_folder + '/*'): 
#             original_images[f'frame{fr_num}.jpg'] = path.split('/')[-1]
#             frame = imageio.imread(path)
#             imageio.imwrite(f'./Zero_DCE/data/test_dataset/frame{fr_num}.jpg', frame)
#             fr_num+=1 

    if args.video_folder is not None :
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


#     t1 = time.time()
#     print('Writing files time: ', t1-t0)
   
# # LLE
#     images = [os.path.join('./Zero_DCE/data/test_dataset', x) for x in os.listdir('./Zero_DCE/data/test_dataset')]
#     for img_path in images:
#     #   print(img_path)
#       img = cv2.imread(img_path)
#       brightness = brightness_level(img)
#       if brightness == 'bright':
#         img = clahe(img)
#         cv2.imwrite(img_path.replace('test_dataset', 'resultset') ,img)
#       elif brightness == 'dark':
#         lowlight(img_path)
#       else:
#         shutil.copy(img_path, img_path.replace('test_dataset', 'resultset'))

    t2 = time.time()
    # print("Image Enhancer: ", t2-t1)
    

    t3 = time.time()
    

    os.chdir('./Dsfd')
    ag = dsfd('../Zero_DCE/data/resultset')
    os.chdir('..')

    t4 = time.time()
    print('DSFD : ', t4 - t3)

    #GFPGAN
    # torch.cuda.empty_cache()
    # os.makedirs(args.output, exist_ok=True)
    # os.chdir('./GFPGAN')
    # args_gfp = Args_gfp(input='../cropped_faces',
    #                     output='../results/', suffix='out')
    # gfpgan(args_gfp)
    # os.chdir('..')
    t5 = time.time()
    # print("GFPGAN: ", t5-t4)


    age_model = init_age_model()
    ort_sess = init_gender_model()

    images = [os.path.join('./results/restored_imgs', x) for x in os.listdir('./results/restored_imgs')]
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
    if args.image_folder is not None:
        ag.to_csv(os.path.join(args.output,"results.csv"), index = False)
    else:
        ag.to_csv(os.path.join(args.output,video_path.split('/')[-1][:-3] + 'csv'), index = False)
    t6 = time.time()
    # print(f'Finished results in {args.output}')   
    print('Age and gender: ', t6-t5)  
    print('Total time: ', t6-t0)   


if __name__ == '__main__':
    main()
