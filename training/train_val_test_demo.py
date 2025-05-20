import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from sklearn import metrics
from data.data_generators import dataloader


class train_val_test_demo():
    def __init__(self, model, model_path, method, epochs=200, lr=0.0001, use_gpu=True):
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.model = model.to(self.device)
        self.model_path = model_path
        model_last_path = model_path.replace('.pth.tar', 'last.pth.tar')
        self.model_last_path = model_last_path
        self.epochs = epochs
        self.lr = lr
        self.method = method
        self.metrics = 'ACC'
        # self.metrics = 'AUC'
        # self.criterion = nn.BCELoss().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.best_auc = 0
        self.best_acc = 0
        self.epoch_start = 0
        self.optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0.)
        self.cos = False
        self.schedule = [5, 10, 15, 20]
        self.history = {'epoch': [], 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'val_auc': []}

    def fit(self, training_filenames, train_params_singleview, validation_filenames, validation_params_singleview):
        epochs = self.epochs
        epoch_start = self.epoch_start

        # training loop
        for epoch in range(epoch_start, epochs + 1):

            train_loader = dataloader(training_filenames, **train_params_singleview)
            valid_loader = dataloader(validation_filenames, **validation_params_singleview)

            train_loss, train_acc = self.__train__(train_loader, epoch)
            val_loss, val_acc, val_auc = self.__val__(valid_loader)

            self.history['epoch'].append(epoch)
            self.history['loss'].append(train_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_auc'].append(val_auc)

            if self.metrics == 'ACC':
                is_best = val_acc > self.best_acc
                self.best_acc = max(val_acc, self.best_acc)
                self.best_auc = val_auc
            else:
                is_best = val_auc > self.best_auc
                self.best_auc = max(val_auc, self.best_auc)
                self.best_acc = val_acc

            if is_best:
                print('epoch:', epoch, 'is test best now, will save in csv, ACC is :', val_acc, ', AUC is :', val_auc)

            if is_best:
                torch.save({'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'best_acc': self.best_acc,
                            'best_auc': self.best_auc,
                            'optimizer': self.optimizer.state_dict(),
                            'history': self.history},
                           self.model_path)

            torch.save({'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'best_acc': self.best_acc,
                        'best_auc': self.best_auc,
                        'optimizer': self.optimizer.state_dict(),
                        'history': self.history},
                       self.model_last_path)

        return self.history

    def __train__(self, train_loader, epoch):
        self.model.train()
        self.adjust_learning_rate(epoch)
        total_loss, total_top1, total_num, train_bar = 0.0, 0.0, 0, tqdm(train_loader)
        for input in train_bar:
            img = input['img']
            label = input['label']
            # label = label.unsqueeze(1)
            img, label = img.to(self.device), label.to(self.device)
            # model
            pred = self.model(img)
            # compute loss
            loss = self.criterion(pred, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pred = F.softmax(pred, dim=1)
            total_loss += loss.item() * train_loader.batch_size
            total_num += img.size(0)
            _, predicted = torch.max(pred.data, 1)
            total_top1 += (predicted == label).float().sum().item()

            train_bar.set_description(
                'TraEpo:[{}/{}],'
                'lr:{:.4f},'
                'TraLos:{:.2f},'
                'TraAcc:{:.2f}'.format(epoch,
                                       self.epochs,
                                       self.optimizer.param_groups[0]['lr'],
                                       total_loss / total_num,
                                       total_top1 / total_num * 100))

        return total_loss / total_num, total_top1 / total_num

    def __val__(self, valid_loader):
        self.model.eval()
        all_probabilities, all_predicted, all_labels = [], [], []
        total_loss, total_top1, total_num = 0.0, 0.0, 0
        with torch.no_grad():
            valid_bar = tqdm(valid_loader)
            for input in valid_bar:
                img = input['img']
                label = input['label']
                img, label = img.to(self.device), label.to(self.device)
                # model
                pred = self.model(img)
                all_labels.append(label.data.cpu().numpy())
                # compute loss
                loss = self.criterion(pred, label)
                total_loss += loss.item() * valid_loader.batch_size
                total_num += img.size(0)
                pred = F.softmax(pred, dim=1)
                all_probabilities.append(pred.data.cpu().numpy())
                _, predicted = torch.max(pred.data, 1)
                all_predicted.append(predicted.cpu().numpy())
                total_top1 += (predicted == label).float().sum().item()

                valid_bar.set_description(
                    'Val Loss: {:.2f},'
                    'Acc:{:.2f}%'.format(total_loss / total_num, total_top1 / total_num * 100))

            y_true = all_labels
            y_true = np.array(y_true)
            y_true = y_true.reshape(-1, 1)
            y_pred = all_predicted
            y_pred = np.array(y_pred)
            y_pred = y_pred.reshape(-1, 1)
            cm = metrics.confusion_matrix(y_true, y_pred)
            print(cm)
            all_labels_np = np.array(all_labels)
            all_probabilities_np = np.array(all_probabilities)
            all_labels_np = all_labels_np.reshape(-1, 1)
            all_probabilities_np = all_probabilities_np.reshape(-1, 2)
            all_probabilities_np = all_probabilities_np[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(all_labels_np, all_probabilities_np)
            roc_auc = metrics.auc(fpr, tpr)

        return total_loss / total_num, total_top1 / total_num, roc_auc

    def test(self, test_loader):
        self.model.eval()
        all_predicted = []
        total_loss, total_top1, total_num = 0.0, 0.0, 0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for input in test_bar:
                img = input['img']
                label = input['label']
                img, label = img.to(self.device), label.to(self.device)
                # model
                pred = self.model(img)
                # compute loss
                loss = self.criterion(pred, label)
                total_loss += loss.item() * test_loader.batch_size
                total_num += img.size(0)
                pred = F.softmax(pred, dim=1)
                all_predicted.append(pred.data.cpu().numpy())
                _, predicted = torch.max(pred.data, 1)
                total_top1 += (predicted == label).float().sum().item()

                test_bar.set_description(
                    'test Loss: {:.4f}, '
                    'Acc:{:.2f},%'.format(total_loss / total_num, total_top1 / total_num * 100))

        return np.array(all_predicted)

    def adjust_learning_rate(self, epoch):
        """Decay the learning rate based on schedule"""
        lr = self.lr
        if self.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / self.epochs))
        else:  # stepwise lr schedule
            for milestone in self.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch_start = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        self.best_auc = checkpoint['best_auc']
        self.history = checkpoint['history']

