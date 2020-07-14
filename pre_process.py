import os
import pickle
import glob
import hashlib
from types import LambdaType

import cv2 as cv
import mxnet as mx
from mxnet import recordio
from tqdm import tqdm

from config import get_config

from mtcnn import MTCNN
from PIL import Image
from pathlib import Path


mtcnn = MTCNN()
print('mtcnn loaded')

# /BASE_DATA/
#     /0001
#         1.jpg
#         2.jpg
#     /0002
#         1.jpg
#         2.jpg

# anhdh, 1



def clear_path(paths):
    image_type = [".png", ".jpg", ".jpeg"]
    return [path for path in paths if os.path.splitext(path)[1] in image_type]


def create_pickle(detect_face = False):
    conf = get_config()
    samples_train = []
    samples_val = []
    class_ids = {}
    current_class_id = 0

    list_dir = glob.glob(f'{conf.raw_data}/*')
    print(f"[INFO] Num actors: {len(list_dir)}")
    dataset_name = conf.dataset_name
    new_dataset_dir = os.path.join(conf.processed_data, dataset_name)
    
    if not os.path.isdir(new_dataset_dir):
        Path(new_dataset_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(new_dataset_dir):
        Path(conf.train_path).mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(new_dataset_dir):
        Path(conf.val_path).mkdir(parents=True, exist_ok=True)
    # return



    len_data = len(list_dir)
    len_val = int(len_data * conf.dataset_ratio_val)
    len_train = len_data - len_val

    print(f"[INFO] Len data: {len_data}")
    print(f"[INFO] Len train data: {len_train}")
    print(f"[INFO] Len val data: {len_val}")

    print("[INFO] Extracting data for training.")
    for parent_dir in tqdm(list_dir[:len_train]):
        new_samples, label = extract_data(parent_dir, current_class_id, conf.train_path, detect_face)
        samples_train = samples_train + new_samples
        class_ids[current_class_id] = label
        if len(new_samples) == 0:
            continue
        current_class_id = current_class_id + 1

    print("[INFO] Extracting data for val.")
    for parent_dir in tqdm(list_dir[len_train:]):
        new_samples, label = extract_data(parent_dir, current_class_id, conf.val_path, detect_face)
        samples_val = samples_val + new_samples
        class_ids[current_class_id] = label
        if len(new_samples) == 0:
            continue
        current_class_id = current_class_id + 1    

    print(f"[INFO] Num sample train: {len(samples_train)}")
    print(f"[INFO] Num sample val: {len(samples_val)}")
    print(f"[INFO] Num class: {len(class_ids)}")

    with open(conf.pickle_train_images, 'wb') as file_train_images:
        pickle.dump(samples_train, file_train_images)

    with open(conf.pickle_val_images, 'wb') as file_val_images:
        pickle.dump(samples_val, file_val_images)

    with open(conf.pickle_class_labels, "wb") as file_labels:
        pickle.dump(class_ids, file_labels)

def extract_data(parent_dir, current_class_id, new_dataset_dir, detect_face=False):
    if (os.path.isdir(parent_dir)):
        new_samples, label = process_folder_label(current_class_id, new_dataset_dir, parent_dir, detect_face = detect_face)
        
        return new_samples, label
    return [], None

def process_folder_label(current_class_id, new_dataset_dir, parent_dir, detect_face = False):
    conf = get_config()
    samples = []

    # Create labels
    label = os.path.split(parent_dir)[1]
    
    id_label = current_class_id
    
    # print(f"[INFO] create id: {id_label}")
    # Store label and corresponding id

    # new_parent_dir = os.path.join(new_dataset_dir, str(id_label))
    new_parent_dir = os.path.join(new_dataset_dir, str(id_label))

    # print("Dir: ", label)
    for imageFile in clear_path(glob.glob(os.path.join(conf.raw_data, parent_dir, "*"))):
        # image = cv.imread(imageFile)
        # print(f"Image shape: {image.shape}")
        image = Image.open(imageFile).convert('RGB')
        if detect_face:   
            image = mtcnn.align_and_take_one(image)
        if image == None:
            continue
        # image = image_resize(image, new_width=112, new_height=112)
        image= image.resize((112, 112), Image.ANTIALIAS)
        
        if not os.path.isdir(new_parent_dir):
            # os.makedirs(new_parent_dir)
            Path(new_parent_dir).mkdir(parents=True, exist_ok=True)

        nameImage = os.path.split(imageFile)[1]
        
        new_image_path = os.path.join(new_parent_dir, nameImage)
        # cv.imwrite(new_image_path, image)
        image.save(new_image_path)
        line = {"img": new_image_path, "label": id_label}

        samples.append(line)
        
        # print(f"Line data: {line}")
    return samples, label

# BASE_DATA/
#     1.jpg
#     2.jpg

def create_pickle_type2(num_sample=-1):
    conf = get_config()
    samples = []
    class_ids = set()
    count_img = 0
    # print(os.listdir(conf.raw_data))
    print(f'{conf.raw_data}/*.jpg')
    # print(f'{conf.raw_data}/*.jpg')
    arr_path = glob.glob(f'{conf.raw_data}/*.jpg')
    print(f"[INFO] num data: {len(arr_path)}")
    arr_path.sort(key=lambda x: int(os.path.basename(x).replace(".jpg", "")))
    
    for path in tqdm.tqdm(arr_path[:10000]):
        filename = os.path.basename(path)
        label = filename.replace(".jpg", "")

        if (num_sample != -1 and count_img >= num_sample):
            break
        # print(path, label)
        image = cv.imread(path)
        image = image_resize(image, new_width=112, new_height=112)

        if not os.path.isdir(conf.processed_data):
            os.mkdir(conf.processed_data)
        
        new_image_path = os.path.join(conf.processed_data, filename)
        cv.imwrite(new_image_path, image)

        line = {"img": filename, "label": int(label)}
        count_img = count_img + 1

        samples.append(line)
        class_ids.add(label)
    
    print(f"[INFO] Num sample: {len(samples)}")
    print(f"[INFO] Num class: {len(class_ids)}")
    # print(samples)
    with open(conf.pickle_path_images, 'wb') as file:
        pickle.dump(samples, file)

    print('num_samples: ' + str(len(samples)))

    class_ids = list(class_ids)
    print(len(class_ids))
    print(max(class_ids))



def image_resize(image, new_width = None, new_height = None, inter = cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if new_width is None and new_height is None:
        return image

    # check to see if the new_width is None
    if new_width is None:
        # calculate the ratio of the new_height and construct the
        # dimensions
        r = new_height / float(h)
        dim = (int(w * r), new_height)

    # otherwise, the new_height is None
    elif new_height is None:
        # calculate the ratio of the new_width and construct the
        # dimensions
        r = new_width / float(w)
        dim = (new_width, int(h * r))
    
    else:
        dim = (new_width, new_height)
    
    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

if __name__ == "__main__":
    # create_pickle_base()
    create_pickle()
    # /media/ailab/DATA/facescrub/actors/images/Adam_McKay