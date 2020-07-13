import os
import pickle
import glob
import hashlib

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


def create_pickle_type1(num_sample=-1):
    conf = get_config()
    samples = []
    class_ids = {}
    count_img = 0
    current_class_id = 0

    list_dir = glob.glob(f'{conf.raw_data}/*')
    print(f"[INFO] Num actors: {len(list_dir)}")
    dataset_name = os.path.split(conf.raw_data)[1]
    new_dataset_dir = os.path.join(conf.processed_data, dataset_name)
    
    if not os.path.isdir(new_dataset_dir):
        Path(new_dataset_dir).mkdir(parents=True, exist_ok=True)
    # return
   

    for parent_dir in tqdm(list_dir):
        if (os.path.isdir(parent_dir)):
            # Create labels
            label = os.path.split(parent_dir)[1]
            
            id_label = current_class_id
            current_class_id = current_class_id + 1
            # print(f"[INFO] create id: {id_label}")
            # Store label and corresponding id

            # new_parent_dir = os.path.join(new_dataset_dir, str(id_label))
            new_parent_dir = os.path.join(new_dataset_dir, str(id_label))

            # print("Dir: ", label)
            for imageFile in clear_path(glob.glob(os.path.join(conf.raw_data, parent_dir, "*"))):
                if (num_sample != -1 and count_img == num_sample):
                    break
                count_img = count_img + 1
              

                # image = cv.imread(imageFile)
                # print(f"Image shape: {image.shape}")
                image = Image.open(imageFile).convert('RGB')
                image = mtcnn.align(image)
                if image == None:
                    continue
                # image = image_resize(image, new_width=112, new_height=112)
                image= image.resize((112, 112), Image.ANTIALIAS)
                
                if not os.path.isdir(new_parent_dir):
                    os.mkdir(new_parent_dir)

                nameImage = os.path.split(imageFile)[1]
                
                new_image_path = os.path.join(new_parent_dir, nameImage)
                # cv.imwrite(new_image_path, image)
                image.save(new_image_path)
                line = {"img": new_image_path, "label": id_label}
                samples.append(line)
                class_ids[id_label] = label
                # print(f"Line data: {line}")

    print(f"[INFO] Num sample: {len(samples)}")
    print(f"[INFO] Num class: {len(class_ids)}")
    print(class_ids)

    with open(conf.pickle_path_images, 'wb') as file_images:
        pickle.dump(samples, file_images)

    with open(conf.pickle_class_labels, "wb") as file_labels:
        pickle.dump(class_ids, file_labels)

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
    create_pickle_type1()
    # /media/ailab/DATA/facescrub/actors/images/Adam_McKay