import torch
import torch.nn as nn
# from convlstm import convLSTM
# from ConvLSTM_pytorch.convlstm import ConvLSTM


class SSRWEncoderDecoder(nn.Module):
    def __init__(self,
                 hidden_dim_encoder: int=128,
                 hidden_dim_decoder: int=128,
                 vocab_size: int=120,
                 max_len: int=250,
                 img_size: int=64,
                 device=torch.device('cuda:0')):
        super(SSRWEncoderDecoder, self).__init__()
        self.hidden_dim_encoder = hidden_dim_encoder
        self.hidden_dim_decoder = hidden_dim_decoder
        self.max_len = max_len
        self.img_size = img_size 
        self.vocab_size = vocab_size
        self.device = device
        # input (BATCH_SIZE, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # conv1 (BATCH_SIZE, 32, 64, 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        # conv2 (BATCH_SIZE, 64, 64, 64)
        self.pool = nn.MaxPool2d(2, 2)
        # pool1 (BATCH_SIZE, 64, 32, 32)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # conv3 (BATCH_SIZE, 64, 32, 32)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # conv4 (BATCH_SIZE, 64, 32, 32)
        # pool2 (BATCH_SIZE, 64, 16, 16)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # conv5 (BATCH_SIZE, 64, 16, 16)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        # conv5 (BATCH_SIZE, 64, 16, 16)
        self.norm6 = nn.BatchNorm2d(16)
        # pool3 (BATCH_SIZE, 64, 8, 8)
        self.lstm_encoder = Encoder(64*8*8, self.hidden_dim_encoder)
        self.lstm_decoder = Decoder(self.vocab_size, 64*8*8, self.hidden_dim_decoder)
    
    def forward(self, x_seq, h):
        # (seq, BATCH, channel, width, height) へ変更
        x_seq = x_seq.reshape(x_seq.shape[1],
                              x_seq.shape[0],
                              x_seq.shape[2],
                              x_seq.shape[3], 
                              x_seq.shape[4])
        x_batch = []
        for x in x_seq:
            x = self.conv1(x)
            x = self.relu(self.norm1(x))
            x = self.conv2(x)
            x = self.relu(self.norm2(x))
            x = self.pool(x)
            x = self.conv3(x)
            x = self.relu(self.norm2(x))
            x = self.conv4(x)
            x = self.relu(self.norm2(x))
            x = self.pool(x)
            x = self.conv5(x)
            x = self.relu(self.norm1(x))
            x = self.pool(x)
            x = self.conv6(x)
            x = self.relu(self.norm6(x))
            x_batch.append(x)
        x = torch.Tensor(x_batch, device=self.device)
        # (BATCH, seq, channel*width*height) へ変更
        x = x.reshape(x.shape[1],
                      x.shape[0],
                      x.shape[2],
                      x.shape[3],
                      x.shape[4])
        x = x.reshape(x.shape[0],
                      x.shape[1],
                      x.shape[2]*x.shape[3]*x.shape[4])
        h = self.lstm_encoder(x, h)
        x = x[:, :, -1]
        x, h = self.lstm_decoder(x, h)
        return x, h

class Encoder(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    
    def forward(self, x, state):
        _, state = self.lstm(x, state)
        return state
       
class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, state):
        x, state = self.lstm(x, state)
        x = self.linear(x)
        return x, state
