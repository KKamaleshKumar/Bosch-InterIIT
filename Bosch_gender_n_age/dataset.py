from operator import index
import torch
from dla import dla34
import cv2
import onnxruntime as ort
import os
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np

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
        #img = np.transpose(img, (2,0,1))

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5946, 0.4548, 0.3891], std=[0.2191, 0.1949, 0.1856])
            ])

        img = transform(img)
        # print(img.shape)
        img = img.to(device)
        model.eval()
    return model(img.unsqueeze(0)).item()

def gender_detect(img,ort_sess):
        img = cv2.resize(img, (384,384))
        ort_inputs = {ort_sess.get_inputs()[0].name: np.array(img).astype(np.float32)[np.newaxis, ...]}
        outputs = ort_sess.run(None, ort_inputs)
        return outputs[0][0].argmax(0)

if __name__ == '__main__':
    age_model = init_age_model()
    ort_sess = init_gender_model()
    images = [os.path.join('./dataset/content/MyDrive/MyDrive/inter_iit_dataset/', x) for x in os.listdir('./dataset/content/MyDrive/MyDrive/inter_iit_dataset/')]
    df = pd.DataFrame(columns= ['image', 'age', 'gender'])
    for i, img_path in enumerate(tqdm(images)):
      img = cv2.imread(img_path)
      img_path = img_path.split('/')[-1]
      gender = 'F' if gender_detect(img, ort_sess) == 0 else 'M'
      img = cv2.resize(img,(256,256))
      age = age_regression(img, age_model)
      dummy = {'image' : img_path, 'age' : age, 'gender': gender}
      dummy = pd.DataFrame(dummy, index=[0])
      df = pd.concat([df, dummy], ignore_index=True)
    df.to_csv('./dataset.csv', index= False)
    