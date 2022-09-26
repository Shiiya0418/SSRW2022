import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import pickle
from PIL import Image
from image_preprcess import transforms_maker
import csv
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 dataloader_train,
                 dataloader_test,
                 criterion,
                 loss_function,
                 device: torch.device=torch.device('cuda:0')):
        self.device = device
        self.model = model.to(device=device)
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.criterion = criterion
        self.loss_function = loss_function
    
    def detach(self, states):
        return [states.detach() for state in states]
    
    def train(self, epochs: int=10, batch_size: int=8, hidden_dim:int=128):
        self.train_loss_list =[]
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []
        for epoch in tqdm(range(epochs), total=epochs):
            train_loss = 0.0
            train_acc = 0.0
            test_loss = 0.0
            test_acc = 0.0

            states = (torch.zeros(1, batch_size, hidden_dim).to(self.device),
                    torch.zeros(1, batch_size, hidden_dim).to(self.device))
            self.model.train()
            print('train')
            for i, batch in enumerate(self.dataloader_train):
                print(f'train: {i}/{len(self.dataloader_train)}')

class SpeakingDataset(Dataset):
    def __init__(self, 
                 input_size: int=64,
                 train_path: str='train/transformed_train',
                 labels_path: str='train/labels',
                 data_prefix: str='ROHAN4600_',
                 dict_path: str='train/label.pkl',
                 is_test: bool=False):
        super().__init__()
        
        self.input_size = input_size
        self.train_path = train_path
        self.labels_path = labels_path
        self.data_prefix = data_prefix
        self.dict_path = dict_path
        self.is_test = is_test
        self.transform = transforms_maker()
        # 全発話ディレクトリ、音素アノテーションをロード
        self.train_dir_list = sorted(glob.glob(self.train_path + '/' + self.data_prefix + '*'))
        self.labels_list = sorted(glob.glob(self.labels_path + '/*'))
        # 音素辞書をロード
        with open(self.dict_path, 'rb') as f:
            self.dict = pickle.load(f)
        
    
    def __len__(self):
        return len(self.train_dir_list)
    
    def __getitem__(self, index: int):
        train_path = self.train_dir_list[index]
        label_path = self.labels_list[index]

        # 入力データ
        images = []
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
        
        return images, tokens
        
def collate_fn(batch: torch.Tensor):
    images, labels = [], []

    for image, label in batch:
        images.append(image)
        labels.append(label)


if __name__ == '__main__':
    dataset = SpeakingDataset()
