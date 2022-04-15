import cv2
import numpy as np
import os
import pandas as pd
import argparse
from empty_directories import empty_directory

from dsfd import empty_directory


def video_generator(video_path = "./input/video1.mp4", save_dir = './', csv_path = './results.csv'):
    video_path = video_path
    saving_path = os.path.join(save_dir, f'result_{video_path.split("/")[-1]}')
    # img_dir = './face_bbox/' 
    if os.path.exists(saving_path):
        os.remove(saving_path)
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_size = (width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    output_video = cv2.VideoWriter(saving_path, fourcc, fps, frame_size, True)
    j = 0
    while cap.isOpened():
        ret, f = cap.read()
        if ret:
            fd = df[df['frame_num'] == j]
            fd = fd.reset_index(drop = True)
            for i in range(len(fd)):
                x0 = int(fd.iloc[i:i+1,  2:3].values)
                y0 = int(fd.iloc[i:i+1,  3:4].values)
                x1 = int(fd.iloc[i:i+1 , 2:3].values + fd.iloc[i:i+1 , 5:6].values)
                y1 = int(fd.iloc[i:i+1 , 3:4].values + fd.iloc[i:i+1 , 4:5].values)
                cv2.rectangle(f, (x0, y0), (x1, y1), (0, 0, 255), 2)
                f = cv2.putText(f, '{:.0f} {}'.format(fd.iloc[i:i+1, 6:7].values[0][0], fd.iloc[i:i+1, 7:8].values[0][0]), (x0-3, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            output_video.write(f)
            j += 1
        else:
            break
    print('Video result saved at: ', saving_path)


    cap.release()
    output_video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_path', type=str,
                        default='./results.csv', help='CSV file containing bbox, gender and age')
    parser.add_argument('-o', '--origin_video', type=str,
                        default='./input/video1.mp4', help='Path of video whose data is stored in csv.')
    parser.add_argument('-s', '--save_dir', default= './', help = 'Directory to store output video.')
    

    args = parser.parse_args()


    video_generator(video_path= args.origin_video, save_dir= args.save_dir, csv_path= args.csv_path)
