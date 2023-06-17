import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import pickle
import numpy

device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
torch.manual_seed(1111)
weights = torch.randn(72, 10).to(device)

all_symb = {'A': 1, 'a': 2, 'B': 3, 'b': 4, 'C': 5, 'c': 6, 'D': 7, 'd': 8, 'E': 9, 'e': 10, 'F': 11, 'f': 12, 'G': 13, 'g': 14, 'H': 15, 'h': 16, 'I': 17, 'i': 18, 'J': 19, 'j': 20, 'K': 21, 'k': 22, 'L': 23, 'l': 24, 'M': 25, 'm': 26, 'N': 27, 'n': 28, 'O': 29, 'o': 30, 'P': 31, 'p': 32, 'Q': 33, 'q': 34, 'R': 35, 'r': 36, 'S': 37, 's': 38, 'T': 39, 't': 40, 'U': 41, 'u': 42, 'V': 43, 'v': 44, 'W': 45, 'w': 46, 'X': 47, 'x': 48, 'Y': 49, 'y': 50, 'Z': 51, 'z': 52, '#': 53, '+': 54, '=': 55, '~': 56, '*': 57, ',': 58, '.': 59, '-': 60, '_': 61, '0': 62, '1': 63, '2': 64, '3': 65, '4': 66, '5': 67, '6': 68, '7': 69, '8': 70, '9': 71}
all_symb_enc = {1: 'A', 2: 'a', 3: 'B', 4: 'b', 5: 'C', 6: 'c', 7: 'D', 8: 'd', 9: 'E', 10: 'e', 11: 'F', 12: 'f', 13: 'G', 14: 'g', 15: 'H', 16: 'h', 17: 'I', 18: 'i', 19: 'J', 20: 'j', 21: 'K', 22: 'k', 23: 'L', 24: 'l', 25: 'M', 26: 'm', 27: 'N', 28: 'n', 29: 'O', 30: 'o', 31: 'P', 32: 'p', 33: 'Q', 34: 'q', 35: 'R', 36: 'r', 37: 'S', 38: 's', 39: 'T', 40: 't', 41: 'U', 42: 'u', 43: 'V', 44: 'v', 45: 'W', 46: 'w', 47: 'X', 48: 'x', 49: 'Y', 50: 'y', 51: 'Z', 52: 'z', 53: '#', 54: '+', 55: '=', 56: '~', 57: '*', 58: ',', 59: '.', 60: '-', 61: '_', 62: '0', 63: '1', 64: '2', 65: '3', 66: '4', 67: '5', 68: '6', 69: '7', 70: '8', 71: '9'}




with open("/kaggle/input/weight/symbs.pickle", "rb") as f:
    genname = pickle.load(f)

with open("/kaggle/input/weight/dataset2.pickle", "rb") as f:
    dataset = pickle.load(f)



def decode_from(data):
    data = data.detach().cpu().numpy()
    enc_str = ""
    for i in data:
        if int(i) > 71 or int(i) < 0:
            enc_str = enc_str + "<ERR_TOK>"
        elif int(i) == 0:
            continue
        else:
          enc_str = enc_str + all_symb_enc[int(i)]
    return enc_str


def encode_in_tns(inpt, strlen):
    d1_len = 500
    d1_list = []
    if len(inpt)>strlen:
        while len(inpt)>strlen:
            inpt = inpt[:-1]
    
    for i in inpt:
        d1_list.append(all_symb[i])
    if len(d1_list) < d1_len:
        while len(d1_list) < d1_len:
            d1_list.append(0)
#         d2_list = []
#         d2_list.append(all_symb[i])
#         while len(d2_list) < d2_len:
#             d2_list.append(0)
#         d1_list.append(d2_list)
#     if len(d1_list) < d1_len:
#         while len(d1_list) < d1_len:
#             d2_list = []
#             d2_list.append(60)
#             while len(d2_list) < d2_len:
#                 d2_list.append(0)
#             d1_list.append(d2_list)
    return d1_list





# def output_enc(inp_list):
#     all_len = 500
#     fin_list = []
#     for charct in inp_list:
#             fin_list.append(all_symb[charct])
#     if len(fin_list) < all_len:
#         while len(fin_list) < all_len:
#             fin_list.append(60)
#     return fin_list


# class Embedng(nn.Module):
#     def __init__(self):
#         super(Embedng, self).__init__()
#         # self.vocab = list(all_symbs)
        
#         self.weights = weights
#         self.embed = nn.Embedding.from_pretrained(weights).to(device)
#         self.cos = nn.CosineSimilarity(dim=2)

#     def to_embed_seq(self, seqs):
#         seqs = torch.IntTensor(seqs).to(device)
#         emb_seq = self.embed(seqs)
#         return emb_seq

#     def unembed(self, embedded_sequence):
#         weights = self.embed.state_dict()['weight']
#         weights = weights.transpose(0, 1).unsqueeze(0).unsqueeze(0)
#         e_sequence = embedded_sequence.unsqueeze(3).data
#         cosines = self.cos(e_sequence, weights)
#         _, indexes = torch.topk(cosines, 1, dim=2)
# #         words = []
# #         for word in indexes:
# #             word_l = ''
# #             for char_index in word:
# #                 if char_index == 0:
# #                     continue
# #                 else:
# #                     word_l += all_symb_enc[int(char_index)]
# #             if word_l != '':
# #                 words.append(word_l)
# #             else:
# #                 continue
#         return indexes


# embedding = Embedng()
# embedding = embedding.to(device)


class RepackNet(nn.Module):
    def __init__(self):
        super(RepackNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = [10, 12], stride = 2, padding = 2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=[6,7], stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = [10,12], stride = 2, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = [6,7], stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = [8,10], stride = 2, padding = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = [6,4], stride = 1))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = [8,10], stride = 2, padding = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = [4,2], stride = 2))
        self.dropout = nn.Dropout()
        self.layer5 = nn.LSTM(44544, 500, 5)
        
        
#         self.inp = nn.Linear(inp_size, 7500)
#         self.fc = nn.Linear(7500, 2500)
#         self.fc1 = nn.RNN(2500, 1250, 5, dropout = 0.5)
#         self.fc2 = nn.RNN(1250, 250, 5, dropout = 0.5)
#         self.out = nn.Linear(250, outp_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(1, -1)
        x = self.dropout(x)
        x, _ = self.layer5(x)
        out = x.reshape(-1)
        return out
    
#     def decode(self, x):
#         x = self.decode(x, x)
#         return x
    
    
net = RepackNet()
if device == 'cuda':
    net.cuda()



def resize_to_tns(x):
    filelen = round(len(x)/11200000, 3)
    filelen = round(filelen*500)
    x = numpy.array(x)
    dispatch = 11250000 - len(x)
    delta_11 = numpy.full((dispatch), 0)
    if len(x) < 11250000:
#         while len(x) < 11200000:
            x = numpy.append(x, delta_11)
    x = x.reshape(1, 500, 22500)
    return x, filelen
    numpy.delete(x)



def train():
    net.train()
    optim = torch.optim.SGD(net.parameters(), lr=3)#, momentum = 0.9)
    crit = nn.HuberLoss()
    pfj = []
    for i in range(30):
        pfj.append(i)
    for epoch in range(1000):
        print("Epoch: ", epoch)
        random.shuffle(pfj)
        loss_in_ep = []
        pers_in_ep = []
        for i in pfj:
            optim.zero_grad()
            data1 = genname[i]
            data2, filelen = resize_to_tns(dataset[i])
            inp = torch.tensor(data2, dtype = torch.float32).to(device)
            target = torch.tensor(encode_in_tns(data1, filelen), dtype = torch.float32).to(device)
            #target = embedding.to_embed_seq(encode_in_tns(genname[i]), dtype=torch.float32).to(device)
#             print(target.size())
            outp = net(inp)
#             print(numpy.shape(outp))
            loss = crit(outp, target)
            loss.backward()
            optim.step()
            loss_in_ep.append(loss.item())
            outp = decode_from(outp)
            cur_p = 0
            for sym_o, sym_d in zip(outp, data1):
                if len(sym_o) >= 500:
                    while len(sym_o) >= 500:
                        sym_o = sym_o[:-1]
                if sym_o == sym_d:
                    cur_p += 1
            if cur_p != 0:    
                cur_p = round((len(sym_d)/cur_p)*100.0, 1)
            pers_in_ep.append(round(cur_p, 1))
#             print(outp, len(outp))
        epoch_prs = round(sum(pers_in_ep)/len(pers_in_ep), 1)
        if epoch_prs == 100.0:
                 break
        print("\nPersentage in epoch: ", epoch_prs)
        print("AVG loss: ", sum(loss_in_ep)/len(loss_in_ep))
        print("Last output: ", outp)

try:
   train()
   torch.save(net.state_dict(), 'model_weights.pth')
except KeyboardInterrupt:
   torch.save(net.state_dict(), 'model_weights.pth')
   KeyboardInterrupt()
