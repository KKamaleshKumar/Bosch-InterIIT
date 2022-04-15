import shutil
import os

def handler(func, path, exc_info):
    pass

def empty_directory(dir_path):
    shutil.rmtree(dir_path, onerror=handler)
    os.makedirs(dir_path)
# empty directories
if __name__ == '__main__':
    empty_directory('./Zero_DCE/data/test_dataset')
    empty_directory('./Zero_DCE/data/resultset')
    empty_directory('./Dsfd/images')
    empty_directory('./cropped_faces')
    empty_directory('./results')
    empty_directory('./face_bbox')
