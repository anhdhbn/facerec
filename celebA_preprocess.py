import os
import pickle
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
import numpy as np


def replace_same_chars(string, char):
    pattern = char + '{2,}'
    string = re.sub(pattern, char, string)
    return string


def crop_image(image_path, bbox):
    img = Image.open(image_path).convert('RGB')
    x, y, w, h = bbox
    img = img.crop((x, y, x+w, y+h))
    img = img.resize((112, 112), Image.ANTIALIAS)
    return img


def get_pair_img(info, ids, partition):
    intra = []
    inter = []
    img_path = info['img']
    img_name = os.path.basename(img_path)
    id_path = os.path.dirname(img_path)
    id = info['label']
    if 'eval' in img_path:
        p = 1
    else:
        p = 2
    bool_ids = np.zeros(max(ids.values()), dtype='int32')
    for _img_name, _id in ids.items():

        if _img_name <= img_name or partition[_img_name] != p:
            continue

        _img_path = os.path.join(id_path, _img_name)
        _info = {'img': _img_path, 'label': _id}

        if _id == id:
            intra.append((info, _info))
        else:
            if bool_ids[_id] == 0 and id < _id:
                inter.append((info, _info))
                bool_ids[_id] = 1
    # print('[INFO] inter:', len(inter))
    # print('[INFO] intra:', len(intra))
    return intra, inter


celeb_folder = '/media/ailab/DATA/celeb_data'
anno_folder = os.path.join(celeb_folder, 'Anno')
img_folder = os.path.join(celeb_folder, 'img_celeba')
partition_folder = os.path.join(celeb_folder, 'Eval')
preprocess_folder = 'pre_processed/celeba'
train_preprocess_folder = os.path.join(preprocess_folder, 'train')
val_preprocess_folder = os.path.join(preprocess_folder, 'eval')
test_preprocess_folder = os.path.join(preprocess_folder, 'test')

if not os.path.isdir(preprocess_folder):
    Path(preprocess_folder).mkdir(parents=True, exist_ok=True)
if os.path.isdir(preprocess_folder):
    Path(train_preprocess_folder).mkdir(parents=True, exist_ok=True)
    Path(val_preprocess_folder).mkdir(parents=True, exist_ok=True)


with open(os.path.join(anno_folder, 'identity_CelebA.txt'), 'r') as ids_file:
    ids = [line.replace('\n', '') for line in ids_file.readlines()]
    ids = [replace_same_chars(id, ' ').split(' ') for id in ids]
    ids = {id[0]: int(id[1]) for id in ids}
    ids_file.close()

# print(f'[INFO] total of identities: {len(ids)}')
# print('[INFO] identity of file 000001.jpg: ', ids['000001.jpg'])

with open(os.path.join(anno_folder, 'list_bbox_celeba.txt'), 'r') as bboxes_file:
    bboxes = [line.replace('\n', '') for line in bboxes_file.readlines()]
    bboxes = [replace_same_chars(bbox, ' ').split(' ', 1) for bbox in bboxes]
    bboxes = bboxes[2:]
    bboxes = {bbox[0]: [int(x) for x in bbox[1].split(' ')] for bbox in bboxes}
    bboxes_file.close()

# print(f'[INFO] total of bounding boxes: {len(bboxes)}')
# print('[INFO] bounding box of 000001.jpg: ', bboxes['000001.jpg'])

with open(os.path.join(partition_folder, 'list_eval_partition.txt'), 'r') as partition_file:
    partition = [line.replace('\n', '') for line in partition_file.readlines()]
    partition = [replace_same_chars(p, ' ').split(' ') for p in partition]
    partition = {p[0]: int(p[1]) for p in partition}
    partition_file.close()

# print(f'[INFO total of partition: {len(partition)}')
# print('[INFO] partition of 000001.jpg: ', partition['000001.jpg'])

dict_train = []
intra_pairs_val = []
inter_pairs_val = []
intra_pairs_test = []
inter_pairs_test = []
par_tqdm = tqdm(partition.items())
for img_name, p in par_tqdm:
    # if p!=1:
    #     continue
    image_path = os.path.join(img_folder, img_name)
    img_cropped = crop_image(image_path, bboxes[img_name])

    if p == 0:  # train
        ids_path = os.path.join(train_preprocess_folder, str(ids[img_name]))
        new_image_path = os.path.join(ids_path, img_name)
        info = {'img': new_image_path, 'label': ids[img_name]}
        dict_train.append(info)
    else:
        if p == 1:  # evaluate
            ids_path = os.path.join(val_preprocess_folder, str(ids[img_name]))
        else:       # test
            ids_path = os.path.join(test_preprocess_folder, str(ids[img_name]))

        new_image_path = os.path.join(ids_path, img_name)
        info = {'img': new_image_path, 'label': ids[img_name]}
        intra, inter = get_pair_img(info, ids, partition)

        if p == 1:
            intra_pairs_val += intra
            inter_pairs_val += inter
            par_tqdm.set_description(f'intra: {len(intra_pairs_val)}')
            par_tqdm.set_description(f'inter: {len(inter_pairs_val)}')
        else:
            intra_pairs_test += intra
            inter_pairs_test += inter

    if not os.path.isdir(ids_path):
        Path(ids_path).mkdir(parents=True, exist_ok=True)

    img_cropped.save(new_image_path)

list_ids_train = os.listdir(train_preprocess_folder)
labels = {id: label for label, id in enumerate(list_ids_train)}


# inter_pairs_val = random.sample(inter_pairs_val, len(intra_pairs_val)*5)
# inter_pairs_test = random.sample(inter_pairs_test, len(intra_pairs_test)*5)

dict_train_path = os.path.join(preprocess_folder, 'train_celeb_112x112.pickle')
intra_pairs_val_path = os.path.join(
    preprocess_folder, 'val_celeb_112x112_intra.pickle')
inter_pairs_val_path = os.path.join(
    preprocess_folder, 'val_celeb_112x112_inter.pickle')
intra_pairs_test_path = os.path.join(
    preprocess_folder, 'test_celeb_112x112_intra.pickle')
inter_pairs_test_path = os.path.join(
    preprocess_folder, 'test_celeb_112x112_inter.pickle')
labels_train_path = os.path.join(preprocess_folder, 'labels_train.pickle')
with open(dict_train_path, 'wb') as pkl_file:
    pickle.dump(dict_train, pkl_file)
    pkl_file.close()

with open(intra_pairs_val_path, 'wb') as pkl_file:
    pickle.dump(intra_pairs_val, pkl_file)
    pkl_file.close()

with open(inter_pairs_val_path, 'wb') as pkl_file:
    pickle.dump(inter_pairs_val, pkl_file)
    pkl_file.close()

with open(intra_pairs_test_path, 'wb') as pkl_file:
    pickle.dump(intra_pairs_test, pkl_file)
    pkl_file.close()

with open(inter_pairs_test_path, 'wb') as pkl_file:
    pickle.dump(inter_pairs_test, pkl_file)
    pkl_file.close()

with open(labels_train_path, 'wb') as pkl_file:
    pickle.dump(labels, pkl_file)
    pkl_file.close()
