import bcolz
import math
from torchvision import transforms as trans
from torchvision.utils import make_grid
from PIL import Image
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from torch import optim
import torch
from verifacation import evaluate, evaluate_custom
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from data.data_pipe import de_preprocess, get_train_loader, get_val_loader, get_val_data, get_val_pair
plt.switch_backend('agg')


class face_learner(object):
    def __init__(self, conf, inference=False, val_custom=True):
        print(conf)
        self.conf = conf
        self.val_custom = val_custom
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(
                conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(
                conf.net_mode, conf.net_depth))

        if not inference:
            self.milestones = conf.milestones
            self.train_loader, self.class_num = get_train_loader(conf)
            if self.val_custom:
                self.val_loader = get_val_loader(conf)
            else:
                # self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(self.train_loader.dataset.root.parent)
                self.lfw, self.lfw_issame = get_val_pair(conf.emore_folder, 'lfw')
            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size,
                                classnum=self.class_num).to(conf.device)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [
                        paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn +
                        [self.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('[INFO] Optimizers generated')
            # self.board_loss_every = len(self.train_loader)//100
            self.board_loss_every = 5
            if (self.board_loss_every < 5):
                self.board_loss_every = 5
            print(f"[INFO] Board loss every: {self.board_loss_every}")
            self.evaluate_every = len(self.train_loader)//2
            print(f"[INFO] Evaluate every: {self.evaluate_every}")
            self.save_every = len(self.train_loader)//4
            print(f"[INFO] Save every: {self.save_every}")
            
        else:
            self.threshold = conf.threshold

    def save_state(self, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = self.conf.save_path
        else:
            save_path = self.conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))

    def load_state(self, fixed_str, from_save_folder=True, model_only=False):
        if from_save_folder:
            save_path = self.conf.save_path
        else:
            save_path = self.conf.model_path
        self.model.load_state_dict(torch.load(
            save_path/'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(
                save_path/'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(
                save_path/'optimizer_{}'.format(fixed_str)))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(
            db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(
            db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(
            db_name), roc_curve_tensor, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)


    def evaluate(self, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), self.conf.embedding_size])
        with torch.no_grad():
            while idx + self.conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + self.conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(
                        batch.to(self.conf.device)) + self.model(fliped.to(self.conf.device))
                    embeddings[idx:idx + self.conf.batch_size] = l2_norm(emb_batch.cpu())
                else:
                    embeddings[idx:idx +
                               self.conf.batch_size] = self.model(batch.to(self.conf.device)).cpu()
                idx += self.conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(
                        batch.to(self.conf.device)) + self.model(fliped.to(self.conf.device))
                    embeddings[idx:] = l2_norm(emb_batch.cpu())
                else:
                    embeddings[idx:] = self.model(batch.to(self.conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(
            embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def find_lr(self,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.train_loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.train_loader), total=num):

            imgs = imgs.to(self.conf.device)
            labels = labels.to(self.conf.device)
            batch_num += 1

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = self.conf.ce_loss(thetas, labels)

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss, batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            # Do the SGD step
            # Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses

    def evaluate_custom(self, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings1 = np.zeros(
            (len(self.val_loader.dataset), self.conf.embedding_size))
        # print(embeddings1.shape)
        embeddings2 =  np.zeros(
            (len(self.val_loader.dataset), self.conf.embedding_size))
        # print(embeddings1.shape)
        actual_issame = np.zeros((len(self.val_loader.dataset)))

        with torch.no_grad():
            samples_tqdm = tqdm(iter(self.val_loader))
            for idx, (imgs1, imgs2, issame) in enumerate(samples_tqdm):

                embedding1 = self.model(imgs1.to(self.conf.device)).cpu().numpy()
                embedding2 = self.model(imgs2.to(self.conf.device)).cpu().numpy()
                # print(f"[INFO] shape embedding1: {embedding1.shape[0]}")
                embeddings1[idx * self.conf.batch_size : idx * self.conf.batch_size + embedding1.shape[0], :] = embedding1

                embeddings2[idx * self.conf.batch_size:idx * self.conf.batch_size + embedding2.shape[0], :] = embedding2
                actual_issame[idx * self.conf.batch_size : idx * self.conf.batch_size + issame.shape[0]] = issame
        # print(f"[INFO] actual_issame : {actual_issame.shape}")
        tpr, fpr, accuracy, best_thresholds = evaluate_custom(
            embeddings1, embeddings2, actual_issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        # print(f"[INFO] Acc: {accuracy.mean()}");
        # print(f"[INFO] TF: {best_thresholds.mean()}");
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor, tpr, fpr

    def train(self, epochs):
        self.model.train()
        running_loss = 0.
        accuracy = 0
        loss_board = 0
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()

            samples_tqdm = tqdm(iter(self.train_loader), position=0, leave=True)
            for imgs, labels in samples_tqdm:
                imgs = imgs.to(self.conf.device)
                # print(f"[INFO] Images: {imgs}")
                labels = labels.to(self.conf.device)
                # print(f"[INFO] Labels: {labels}")
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                # print(f"[INFO] Embeddings: {embeddings}")
                thetas = self.head(embeddings, labels)
                loss = self.conf.ce_loss(thetas, labels)
                # print("size of thetas: ", thetas.size())
                # print(f"[INFO] Loss: {loss}")
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    samples_tqdm.set_description('Epoch: {:.0f}, Step: {:.0f}, Loss: {:.4f}'.format(
                        e, self.step, loss_board
                    ))
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    if self.val_custom:
                        accuracy, best_threshold, roc_curve_tensor, tpr, fpr = self.evaluate_custom()
                        print('\n Loss: {:.4f}, Accuracy: {:.4f}, Best_threshold: {:.4f}'.format(
                            loss_board, accuracy, best_threshold
                            ))
                        self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    else:
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.lfw, self.lfw_issame, nrof_folds=10, tta=True)
                        print('\n Loss: {:.4f}, Accuracy: {:.4f}, Best_threshold: {:.4f}'.format(
                            loss_board, accuracy, best_threshold
                            ))
                        self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)

                    self.model.train()

                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(accuracy)

                self.step += 1
        self.save_state(accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

    def infer(self, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(self.conf.test_transform(
                    img).to(self.conf.device).unsqueeze(0))
                emb_mirror = self.model(self.conf.test_transform(
                    mirror).to(self.conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(self.conf.test_transform(
                    img).to(self.conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - \
            target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
