from torchvision import transforms as trans
from torch.nn import CrossEntropyLoss
import torch
from pathlib import Path
from easydict import EasyDict as edict


def get_config(training=True):
    conf = edict()
    # conf.raw_data = Path("/media/ailab/DATA/FaceImages")
    # conf.raw_data = Path("/media/ailab/DATA/facescrub2")

    conf.dataset_name = "facescrub"
    conf.dataset_ratio_val = 0.2
    # train val 80 20
    conf.processed_data = Path("./pre_processed/")
    conf.train_path = conf.processed_data/f'{conf.dataset_name}'/'train'
    conf.val_path = conf.processed_data/f'{conf.dataset_name}'/'val'

    conf.pickle_train_images = conf.train_path/'faces_ailab_112x112.pickle'
    conf.pickle_val_inter = conf.val_path/'faces_ailab_112x112_inter.pickle'
    conf.pickle_val_intra = conf.val_path/'faces_ailab_112x112_intra.pickle'

    conf.pickle_class_labels = conf.processed_data / \
        f'{conf.dataset_name}'/'faces_ailab_labels.pickle'

    conf.data_path = Path('data')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112, 112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'  # or 'ir'
    conf.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    conf.data_mode = 'celeb'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.batch_size = 100  # irse net depth 50
    conf.batchsize_infer = 32
#   conf.batch_size = 200 # mobilefacenet
# --------------------Training Config ------------------------
    if training:
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [12, 15, 18]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 4
        conf.ce_loss = CrossEntropyLoss()
# --------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10
        # when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30
        # the larger this value, the faster deduction, comes with tradeoff in small faces

    if conf.data_mode == 'celeb':
        conf.dataset_name = "celeba"
        conf.dataset = conf.processed_data/f'{conf.dataset_name}'
        conf.train_path = conf.dataset/'train'
        conf.val_path = conf.dataset/'val'
        conf.pickle_class_labels = conf.dataset/'labels_train.pickle'
        conf.pickle_train_images = conf.dataset/'train_celeb_112x112.pickle'
        conf.pickle_val_inter = conf.dataset/'val_celeb_112x112_inter.pickle'
        conf.pickle_val_intra = conf.dataset/'val_celeb_112x112_intra.pickle'
        conf.work_path = Path('work_space/')
        conf.model_path = conf.work_path/'models'
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'

    return conf
