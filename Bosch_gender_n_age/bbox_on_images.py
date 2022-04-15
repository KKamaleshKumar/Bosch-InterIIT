import enum
import cv2
import numpy as np
import os
import pandas as pd
import argparse
import glob
import imageio


def annotate_images(image_folder_path= './input/video1.mp4', csv_path = './results.csv' ):
    # images = [os.path.join('./Zero_DCE/data/resultset', x) for x in os.listdir('./Zero_DCE/data/resultset')]
    df = pd.read_csv(csv_path)
    os.makedirs('./face_bbox/', exist_ok=True)
    # print(images)
    # print(os.path.join(image_folder_path,"*.jpg"))
    for _,i in enumerate(glob.glob(os.path.join(image_folder_path,"*.jpg"))):
            img = cv2.imread(i)
            j = int(i.split('/')[-1].replace('.jpg', '')[5:])
            fd = df[df['frame_num'] == j]
            fd = fd.reset_index(drop = True)
            # print(fd)
            for i in range(len(fd)):
                x0 = int(fd.iloc[i:i+1,  2:3].values)
                y0 = int(fd.iloc[i:i+1,  3:4].values)
                x1 = int(fd.iloc[i:i+1 , 2:3].values + fd.iloc[i:i+1 , 5:6].values)
                y1 = int(fd.iloc[i:i+1 , 3:4].values + fd.iloc[i:i+1 , 4:5].values)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                # print(fd.iloc[i:i+1, 6:7].values, fd.iloc[i:i+1, 7:8].values)
                img = cv2.putText(img, '{:.0f} {}'.format(fd.iloc[i:i+1, 6:7].values[0][0], fd.iloc[i:i+1, 7:8].values[0][0]), (x0-3, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(f'./face_bbox/frame{j}.jpg', img)
            j+=1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_path', type=str,
                        default='./results.csv', help='CSV file containing bbox, gender and age')
    parser.add_argument('-o', '--origin_folder', type=str,
                        default='./Zero_DCE/data/resultset', help='Path of video whose data is stored in csv.')
    parser.add_argument('-s', '--save_dir', default= './', help = 'Directory to store output video.')
    

    args = parser.parse_args()
    annotate_images(image_folder_path= args.origin_folder, csv_path= args.csv_path)
    # print('Annotated images')
    # print(len([x for x in os.listdir('./face_bbox/')]))
    
