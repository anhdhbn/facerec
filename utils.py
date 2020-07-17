from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io
from torchvision import transforms as trans
from data.data_pipe import de_preprocess
import torch
from model import l2_norm
import pdb
import cv2

import os
import random
from glob import glob
from tqdm import tqdm

from data.data_pipe import InferenceDataset
from torch.utils.data import DataLoader

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

def prepare_facebank(conf, model, mtcnn, tta = True):
    model.eval()
    embeddings =  []
    names = ['Unknown']
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except:
                        continue
                    if img.size != (112, 112):
                        img = mtcnn.align(img)
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:                        
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0,keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path/'facebank.pth')
    np.save(conf.facebank_path/'names', names)
    return embeddings, names

def load_facebank(conf):
    embeddings = torch.load(conf.facebank_path/'facebank.pth')
    names = np.load(conf.facebank_path/'names.npy')
    return embeddings, names

def face_reader(conf, conn, flag, boxes_arr, result_arr, learner, mtcnn, targets, tta):
    while True:
        try:
            image = conn.recv()
        except:
            continue
        try:            
            bboxes, faces = mtcnn.align_multi(image, limit=conf.face_limit)
        except:
            bboxes = []
            
        results = learner.infer(conf, faces, targets, tta)
        
        if len(bboxes) > 0:
            print('bboxes in reader : {}'.format(bboxes))
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice            
            assert bboxes.shape[0] == results.shape[0],'bbox and faces number not same'
            bboxes = bboxes.reshape([-1])
            for i in range(len(boxes_arr)):
                if i < len(bboxes):
                    boxes_arr[i] = bboxes[i]
                else:
                    boxes_arr[i] = 0 
            for i in range(len(result_arr)):
                if i < len(results):
                    result_arr[i] = results[i]
                else:
                    result_arr[i] = -1 
        else:
            for i in range(len(boxes_arr)):
                boxes_arr[i] = 0 # by default,it's all 0
            for i in range(len(result_arr)):
                result_arr[i] = -1 # by default,it's all -1
        print('boxes_arr ： {}'.format(boxes_arr[:4]))
        print('result_arr ： {}'.format(result_arr[:4]))
        flag.value = 0

hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2,
                    (0,255,0),
                    3,
                    cv2.LINE_AA)
    return frame


def clear_non_image(folder_path):
    paths = glob(f"{folder_path}/*")
    image_type = [".png", ".jpg", ".jpeg"]
    return [path for path in paths if os.path.splitext(path)[1] in image_type]

def take_random_image_in_folder(folder_path, take=2):
    list_images = clear_non_image(folder_path)
    return list(set(random.choices(list_images, k=take)))

def take_random_folder_label(paths, take=1, except_folder=""):
    paths = [path for path in paths if path != except_folder]
    return list(set(random.choices(paths, k=take)))

def get_pairs_intra_label(path):
    list_images = clear_non_image(path)
    list1 = list_images[0::2]
    list2 = list_images[1::2]
    return [(list1[idx], list2[idx]) for idx in range(min(len(list1), len(list2)))]

def get_pairs_inter_label(current_path, list_label_folder, intra_pairs):
    # result = []
    # for idx in range(len(intra_pairs)):
    #     img1 = intra_pairs[idx][0]
    #     path = take_random_folder_label(list_label_folder)[0]
    #     img2 = take_random_image_in_folder(path, 1)[0]
    #     result.append((img1, img2))
    # return result
    return [(intra_pairs[idx][0], take_random_image_in_folder(take_random_folder_label(list_label_folder, except_folder=current_path)[0], 1)[0]) for idx in range(len(intra_pairs))]

def get_embed(img_path, learner, conf):
    img = Image.open(img_path).convert('RGB').resize((112, 112), Image.ANTIALIAS)
    # train_transform = trans.Compose([
    #     trans.RandomHorizontalFlip(),
    #     trans.ToTensor(),
    #     trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # ])
    mirror = trans.functional.hflip(img)
    emb = learner.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
    emb_mirror = learner.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
    return l2_norm(emb + emb_mirror)

def findDistance_cos(a, b):
    ''' Compute cosin between 2 vectors
    '''
    # from numpy import dot
    # from numpy.linalg import norm
    # cos_sin = dot(a, b)/(norm(a)*norm(b))
    # return np.arccos(cos_sin)
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    from sklearn.metrics.pairwise import cosine_distances
    return cosine_distances(a, b)[0][0]




def findDistance(a, b):
    ''' Compute distance between 2 vectors
    '''
    return np.linalg.norm(a-b)


def calc_diff_and_similar(all_inter_pairs, all_intra_pairs,conf ,learner, func_find_diff=findDistance_cos, numtake=None):
    similar, diff = [], []
    # for imgs, labels in tqdm(iter(self.train_loader)):
    # all_inter_pairs[:numtake]
    ds_inter = InferenceDataset(all_inter_pairs)
    ds_intra = InferenceDataset(all_intra_pairs)

    loader_inter = DataLoader(ds_inter, batch_size=conf.batchsize_infer, shuffle=True)
    loader_intra = DataLoader(ds_intra, batch_size=conf.batchsize_infer, shuffle=True)
    learner.model.eval()

    for img1s, img2s in tqdm(iter(loader_inter)):
        imgs = img1s.to(conf.device)
        embeddings1 = learner.model(imgs).cpu().detach().numpy()

        imgs = img2s.to(conf.device)
        embeddings2 = learner.model(imgs).cpu().detach().numpy()
        for idx in range(len(embeddings1)):
            diff.append(func_find_diff(embeddings1[idx], embeddings2[idx]))

    for img1s, img2s in tqdm(iter(loader_intra)):
        imgs = img1s.to(conf.device)
        embeddings1 = learner.model(imgs).cpu().detach().numpy()

        imgs = img2s.to(conf.device)
        embeddings2 = learner.model(imgs).cpu().detach().numpy()
        for idx in range(len(embeddings1)):
            similar.append(func_find_diff(embeddings1[idx], embeddings2[idx]))

    return diff, similar
