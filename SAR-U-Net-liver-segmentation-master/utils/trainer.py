import torch
import os
from tensorboardX import SummaryWriter
from matplotlib import pylab as plt
from medpy import metric
from utils.metrics import accuracy
from typing import NamedTuple, List
from visdom import Visdom
import pandas as pd
from tqdm import tqdm
import time

class trainer(object):
    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 dataloaders, comments, verbose_train, verbose_val,
                 ckpt_frequency, max_epochs, checkpoint_dir='checkpoints',
                 start_epoch=0, start_iter=0,alpha=0, device=torch.device('cuda:0'),
                 fold=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.dataloaders = dataloaders
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.max_epochs = max_epochs
        self.verbose_train = verbose_train
        self.verbose_val = verbose_val
        self.ckpt_frequency = ckpt_frequency

        self.epoch = 0
        self.iter = start_iter
        self.alpha = 0.33
        self.total = 0
        self.correct = 0
        self.comments = comments
        self.current_val_loss = 0.0
        self.writer = SummaryWriter(comment=self.comments)
        self.best_val_loss = float('inf')

        # ---- training timer ----
        self._train_time_total = 0.0   # 累積秒數
        self._train_case_total = 0     # 累積影像（case）數  
        self.fold = fold

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def train(self):
        for self.epoch in range(self.max_epochs):
            start = time.time()
            current_lr = self.lr_scheduler.get_lr()
            # lr_list.append(current_lr)
            # epoch_list.append(self.epoch+1)
            print('Epoch: {}'.format(self.epoch + 1))
            print('learning rate: {}'.format(current_lr[-1]))
            epoch_train_loss, epoch_train_acc = self.training_phase(self.dataloaders['train'])
            epoch_val_loss, epoch_val_acc = self.validating_phase(self.dataloaders['val'])
            self.current_val_loss = epoch_val_loss
            if self.current_val_loss < self.best_val_loss:
                self.best_val_loss = self.current_val_loss
                checkpoint_name = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), checkpoint_name)
                print(f'best model saved at {checkpoint_name}')
            print(self.epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc, sep=",")
            self.writer.add_scalar('Tr/Loss(end of epoch)', epoch_train_loss, self.epoch + 1)
            self.writer.add_scalar('Tr/Accurary(end of epoch)', epoch_train_acc, self.epoch + 1)
            self.writer.add_scalar('Val/Loss(end of epoch)', epoch_val_loss, self.epoch + 1)
            self.writer.add_scalar('Val/Accurary(end of epoch)', epoch_val_acc, self.epoch + 1)
            # self.writer.add_scalar('Val/epoch/Loss', epoch_val_loss, self.iter)
            # self.writer.add_scalar('Val/epoch/Accurary', epoch_val_acc, self.iter)
            print('End of the epoch:')
            print('tr loss: {:.16f},tr accurary: {:.4f}'.format(epoch_train_loss, epoch_train_acc))
            print('val loss: {:.16f},val accuracy: {:.4f}'.format(epoch_val_loss, epoch_val_acc))
            self.lr_scheduler.step()
            now = time.time()
            print('Time taken: {:.2f} seconds'.format(now - start))
            print(f'estimated time to finish: {self.max_epochs - self.epoch} epochs')

        self.writer.close()

        #輸出計時結果
        outdir = "training_time/"
        os.makedirs(outdir, exist_ok=True)
        if self._train_case_total:
            sec_per_case = self._train_time_total / self._train_case_total
            outpath = os.path.join(outdir, f"fold{self.fold}.txt" if self.fold is not None else "fold.txt")
            with open(outpath, "w") as f:
                f.write(f"total_seconds\t{self._train_time_total:.6f}\n")
                f.write(f"cases\t{self._train_case_total}\n")
                f.write(f"seconds_per_case\t{sec_per_case:.6f}\n")

    def training_phase(self, dataloader):
        self.model.train()
        loss_total = 0.0
        acc_total = 0.0
        alpha = 0.33
        with tqdm(total=len(dataloader), desc=f'Epoch {self.epoch + 1}', unit='batch') as pbar:
            for inputs, targets, _ in dataloader:
                # print(inputs.shape) torch.Size([4, 1, 256, 256])
                # print(targets.shape) torch.Size([4, 256, 256])
                b = inputs.shape[0]
                self.iter += 1
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    _t0 = time.perf_counter()
                outputs = self.model(inputs.to(self.device))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._train_time_total += time.perf_counter() - _t0
                self._train_case_total +=  b  # 累積 case 數
                loss = self.loss_criterion(outputs, targets.to(self.device))
                # loss1 = self.loss_criterion(outputs[4],targets.to(self.device))
                # loss2 = self.loss_criterion(outputs[3], targets.to(self.device))
                # loss3 = self.loss_criterion(outputs[2], targets.to(self.device))
                # loss4 = self.loss_criterion(outputs[1], targets.to(self.device))
                # loss5 = self.loss_criterion(outputs[0], targets.to(self.device))
                #
                # loss = (loss1+loss2+loss3+loss4) * alpha + loss5

                self.optimizer.zero_grad()
                loss.backward()  # 此时的loss为完成一个iteration的loss
                self.optimizer.step()
                acc = accuracy(outputs, targets)
                loss_total += loss.item() * inputs.size(0)
                acc_total += acc * inputs.size(0)
                if (self.iter % self.verbose_train) == 0:
                    self.writer.add_scalar('Train/iter/Loss', loss.item(),
                                           self.iter)  # loss.item()代表所有iteration(batch)的loss
                    #print('Epoch: {}, iter: {}'.format(self.epoch + 1, self.iter))
                    self.writer.add_scalar('Train/iter/Accurary', acc, self.iter)
                    #print('training loss: {:.16f}, training accuracy: {:.4f}'.format(loss.item(), acc))
                    #tqdm.write('training loss: {:.16f}, training accuracy: {:.4f}'.format(loss.item(), acc))
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

                #if (self.iter % self.verbose_val) == 0:
                #    tqdm.write(f'iter: {self.iter}, start validation')
                #    self.current_val_loss, self.current_acc = self.validating_phase(self.dataloaders['val'])
                #    self.writer.add_scalar('Val/iter/Loss', self.current_val_loss, self.iter)
                #    self.writer.add_scalar('Val/iter/Accurary', self.current_acc, self.iter)
                #    #print('Epoch: {}, iter: {}'.format(self.epoch + 1, self.iter))
                #    #print('val loss: {:.16f}, val accuracy: {:.4f}'.format(self.current_val_loss, self.current_acc))
                #    tqdm.write('validation finished, val loss: {:.16f}, val accuracy: {:.4f}'.format(self.current_val_loss, self.current_acc))
                pbar.update()
        loss_output = loss_total / dataloader.dataset.__len__()
        acc_output = acc_total / dataloader.dataset.__len__()
        return loss_output, acc_output

    def validating_phase(self, dataloader):
        self.model.eval()
        loss_total = 0.0
        acc_total = 0.0
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc=f'Epoch {self.epoch + 1}', unit='batch') as pbar:
                for inputs, targets, _ in dataloader:
                    outputs = self.model(inputs.to(self.device))
                    acc = accuracy(outputs, targets)
                    loss = self.loss_criterion(outputs, targets.to(self.device))
                    loss_total += loss.item() * inputs.size(0)
                    acc_total += acc * inputs.size(0)
                    pbar.update()
        loss_output = loss_total / dataloader.dataset.__len__()
        # acc = metric.dc(outputs, targets.to(self.device))
        acc_output = acc_total / dataloader.dataset.__len__()
        self.model.train()
        return loss_output, acc_output
