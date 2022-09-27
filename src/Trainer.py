import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import pickle
from PIL import Image
from image_preprcess import transforms_maker
import csv
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
import random

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 dataloader_train,
                 dataloader_valid,
                 criterion,
                 optimizer,
                 device: torch.device=torch.device('cuda:0')):
        self.device = device
        self.model = model.to(device=device)
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.criterion = criterion
        self.optimizer = optimizer
    
    def detach(self, states):
        return [state.detach() for state in states]
    
    def train(self, epochs: int=20, batch_size: int=16, hidden_dim:int=128):
        self.train_loss_list =[]
        self.train_acc_list = []
        self.valid_loss_list = []
        self.valid_acc_list = []
        for epoch in tqdm(range(epochs), total=epochs):
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0

            # 初期隠れ状態とせる状態を設定する
            states = (torch.zeros(1, batch_size, hidden_dim).to(self.device),
                    torch.zeros(1, batch_size, hidden_dim).to(self.device))
            # 学習
            self.model.train()
            print('train')
            for i, batch in enumerate(self.dataloader_train):
                print(f'train: {i}/{len(self.dataloader_train)}')
                images, label = batch
                images.to(self.device)
                label.to(self.device)

                self.optimizer.zero_grad()
                states = self.detach(states)
                outputs, states = self.model(images, states)
                label = label.reshape(label.size(0)*label.size(1))
                loss = self.criterion(outputs, label)
                train_loss += loss.item()
                acc = accuracy_score(label.tolist(), outputs.argmax(dim=1).tolist())
                train_acc += acc
                loss.backward()
                self.optimizer.step()
            
            # 検証
            states = (torch.zeros(1, batch_size, hidden_dim).to(self.device),
                    torch.zeros(1, batch_size, hidden_dim).to(self.device))
            self.model.eval()
            print('validation')
            for i, batch in enumerate(self.dataloader_valid):
                print(f'valid: {i}/{len(self.dataloader_valid)}')
                with torch.no_grad():
                    images, label = batch
                    images.to(self.device)
                    label.to(self.device)
                    self.optimizer.zero_grad()
                    states = self.detach(states)
                    outputs, states = self.model(images, states)
                    label = label.reshape(label.size(0)*label.size(1))
                    loss = self.criterion(outputs, label)
                    valid_loss += loss.item()
                    acc = accuracy_score(label.tolist(), outputs.argmax(dim=1).tolist())
                    valid_acc += acc
            
            epoch_loss_train = train_loss / len(self.dataloader_train)
            epoch_acc_train = train_acc / len(self.dataloader_train)
            epoch_loss_valid = valid_loss / len(self.dataloader_valid)
            epoch_acc_valid = valid_acc / len(self.dataloader_valid)
            self.train_loss_list.append(epoch_loss_train)
            self.train_acc_list.append(epoch_acc_train)
            self.valid_loss_list.append(epoch_loss_valid)
            self.valid_acc_list.append(epoch_acc_valid)
            print(f'train: Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss_train:.4f} Acc: {epoch_acc_train:.4f}')
            print(f'valid: Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss_valid:.4f} Acc: {epoch_acc_valid:.4f}')

class DatasetHolder():
    def __init__(self, 
                 input_size: int=64,
                 train_path: str='train/transformed_train',
                 labels_path: str='train/labels',
                 data_prefix: str='ROHAN4600_',
                 dict_path: str='train/label.pkl',
                 train_ratio: int=5,
                 valid_ratio: int=3,
                 test_ratio: int=2):
        self.input_size = input_size
        self.train_path = train_path
        self.labels_path = labels_path
        self.data_prefix = data_prefix
        self.dict_path = dict_path
        self.data_dir_list = sorted(glob.glob(self.train_path + '/' + self.data_prefix + '*'))
        self.labels_list = sorted(glob.glob(self.labels_path + '/*'))
        # 音素辞書をロード
        with open(self.dict_path, 'rb') as f:
            self.dict = pickle.load(f)
        dataset_size = len(self.data_dir_list)
        all_ratio = train_ratio+valid_ratio+test_ratio
        all_indexes = list(range(0, dataset_size))
        train_sample = random.sample(all_indexes,
                                     k=int(dataset_size*(train_ratio/all_ratio)))
        remain_indexes = [e for e in all_indexes if not (e in train_sample)]
        valid_sample = random.sample(remain_indexes,
                                     k=int(dataset_size*(valid_ratio/all_ratio)))
        test_sample = [e for e in remain_indexes if not (e in valid_sample)]

        self.train_list = [dir_path for i, dir_path in enumerate(self.data_dir_list) if i in train_sample]
        self.train_label_list = [dir_path for i, dir_path in enumerate(self.labels_list) if i in train_sample]
        self.valid_list = [dir_path for i, dir_path in enumerate(self.data_dir_list) if i in valid_sample]
        self.valid_label_list = [dir_path for i, dir_path in enumerate(self.labels_list) if i in valid_sample]
        self.test_list = [dir_path for i, dir_path in enumerate(self.data_dir_list) if i in valid_sample]
        self.test_label_list = [dir_path for i, dir_path in enumerate(self.labels_list) if i in test_sample]

        self.dataset_train = SpeakingDataset(phone_dict=self.dict,
                                        dataset_dirs=self.train_list,
                                        label_dirs=self.train_label_list,
                                        is_test=False)
        self.dataset_valid = SpeakingDataset(phone_dict=self.dict,
                                        dataset_dirs=self.valid_list,
                                        label_dirs=self.valid_label_list,
                                        is_test=True)
        self.dataset_test  = SpeakingDataset(phone_dict=self.dict,
                                        dataset_dirs=self.test_list,
                                        label_dirs=self.test_label_list,
                                        is_test=True)
        self.dataset_dict = {'train': self.dataset_train,
                             'valid': self.dataset_valid,
                             'test' : self.dataset_test }
    
    def get_dataset(self, split: str='train'):
        """
        This method will be dataset selected with the argument.
        Please pass the argument with 'train' or 'valid' or 'test'.

        Args:
            split (str, ): _description_. Defaults to 'train'.

        Returns:
            SpeakingDataset: dataset selected with the argument.
        """
        return self.dataset_dict[split]
    
    def get_all_datasets(self) -> list:
        return self.dataset_train, self.dataset_valid, self.dataset_test

    

class SpeakingDataset(Dataset):
    def __init__(self, 
                 phone_dict: dict,
                 dataset_dirs: list,
                 label_dirs: list,
                 input_size: int=64,
                 is_test: bool=False):
        super().__init__()
        
        self.input_size = input_size
        self.dict = phone_dict
        self.is_test = is_test
        self.transform = transforms_maker()
        # 全発話ディレクトリ、音素アノテーションをロード
        self.dataset_dirs = dataset_dirs
        self.label_dirs = label_dirs
        
    
    def __len__(self):
        return len(self.dataset_dirs)
    
    def __getitem__(self, index: int):
        train_path = self.dataset_dirs[index]
        label_path = self.label_dirs[index]

        # 入力データ
        images = []
        # print(f'index: {index} path: {train_path}')
        for image_path in sorted(glob.glob(train_path + '/*')):
            img = Image.open(image_path)
            if self.is_test:
                img = self.transform(img)
            img = np.array(img)
            # to channel first
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)
            images.append(img)
        # ラベルデータ
        with open(label_path, 'r') as f:
            csv_reader = csv.reader(f)
            labels = list(csv_reader)
        tokens = []
        for label in labels[0]:
            tokens.append(self.dict[label])
        images = torch.stack(images, dim=0)
        tokens = torch.Tensor(tokens)
        return images, tokens
        
def collate_fn(batch: torch.Tensor):
    images, label = list(zip(*batch))
    images = pad_sequence(images, batch_first=True)
    label = pad_sequence(label, batch_first=True)
    # print(type(images), type(label))
    return images, label
    """
    max_len_image = max([images.shape(0) for images, _ in batch])
    max_len_label = max([label for _, label in batch])
    images, labels = [], []
    for image, label in batch:
        image = pad_sequence(image, batch_first=True)
        label = pad_sequence(label, batch_first=True)
        images.append(image)
        labels.append(label)
    """


if __name__ == '__main__':
    dataset = SpeakingDataset()
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True,
        collate_fn=collate_fn
    )
    for batch in dataloader:
        print(batch[0].shape)
