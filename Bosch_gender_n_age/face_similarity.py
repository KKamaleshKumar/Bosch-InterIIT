import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class img2feat(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = resnet50(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)

        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, start_dim = 1)
        return x

'''img = torch.rand((1,3,224,224)).to(device)
model = img2feat(output_layer='layer4').to(device)
print(model(img).shape)
'''
def compare_imgs(img1, img2):
    model = img2feat(output_layer = 'layer3').to(device)
    img1 = img1.to(device)
    img2 = img2.to(device)
    v1 = model(img1)
    v2 = model(img2)

    return F.cosine_similarity(v1, v2).item()

t1 = cv2.imread('input/hr1.jpg')
t2 = cv2.imread('input/hr3.jpg')
t3 = cv2.imread('input/hr4.jpg')


t1 = cv2.cvtColor(t1, cv2.COLOR_BGR2RGB)
t2 = cv2.cvtColor(t2, cv2.COLOR_BGR2RGB)
t3 = cv2.cvtColor(t3, cv2.COLOR_BGR2RGB)

t1 = torch.from_numpy(np.transpose(t1, (2,0,1))).float()
t2 = torch.from_numpy(np.transpose(t2, (2,0,1))).float()
t3 = torch.from_numpy(np.transpose(t3, (2,0,1))).float()

t1 = transforms.Resize((224,224))(t1)
t2 = transforms.Resize((224,224))(t2)
t3 = transforms.Resize((224,224))(t3)

t1 = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])(t1)
t2 = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])(t2)
t3 = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])(t3)

t1 = t1.to(device)
t2 = t2.to(device)
t3 = t3.to(device)

#print(t1.unsqueeze(0).shape)
print("Not Similar: ", compare_imgs(t1.unsqueeze(0),t2.unsqueeze(0)))
print("Similar: ", compare_imgs(t1.unsqueeze(0),t3.unsqueeze(0)))
