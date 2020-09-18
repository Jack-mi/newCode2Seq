import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import os
import datetime
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score, ndcg_score, precision_score, recall_score, accuracy_score, median_absolute_error
from sklearn.model_selection import train_test_split, KFold
from mlx_studio.storage import hdfs
from collections import Counter
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, BertConfig, BertForPreTraining

data_path = 'hdfs://haruna/ss_ml/recommend/train_data/reckon_dataset/189551/'
model_path = 'hdfs://haruna/ss_ml/recommend/train_data/reckon_dataset/190358/chinese_L-12_H-768_A-12/'
Is_lock = False
alpha = 15
drop = 0
learning_rate = 2e-5
l2 = 0
batch_size = 128
Input_size = 768
Hidden_size = 768
LR_NUM = 100000

# setting device on GPU if available, else CPU
cuda_gpu = torch.cuda.is_available()
device = torch.device('cuda' if cuda_gpu else 'cpu')
print('Using device:', device)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

# GPU情况
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

config_path = os.path.join(model_path, 'config.json')
checkpoint_path = os.path.join(model_path, 'model.ckpt.index')
vocab_path = os.path.join(model_path, 'vocab.txt')

if not os.path.exists('./bert_model/config.json'):
    hdfs.download('./bert_model/config.json', config_path)
if not os.path.exists('./bert_model/model.ckpt.index'):
    hdfs.download('./bert_model/model.ckpt.index', checkpoint_path)
if not os.path.exists('./bert_model/vocab.txt'):
    hdfs.download('./bert_model/vocab.txt', vocab_path)
    
config_path = './bert_model/config.json'
checkpoint_path = './bert_model/model.ckpt.index'
vocab_path = './bert_model/vocab.txt'

def prob(a, b):
    x = a-b
    return 1.0/(1.0 + torch.exp(-alpha * x))

def prepare_lr_data(optimizer_adam, optimizer_sgd, epochLr, idx):
    df = pd.read_csv('data/good_corpus.csv')[:LR_NUM]
    X = np.array(df['title'])
    Y = np.array(df['ctr'])
    K = 3
    kf = KFold(K, shuffle=True, random_state=22)
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for train_idx, test_idx in kf.split(X):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)
    
    x_train, x_test, y_train, y_test = X_train[idx%K], X_test[idx%K], Y_train[idx%K], Y_test[idx%K]
    train_set = MyDataSet_lr(x_train, y_train)
    test_set = MyDataSet_lr(x_test, y_test)
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    train_step = len(train_dataloader)
    # Model Arguments
    total_steps_2 = epochLr * train_step
    lr_decay_adam = StepLR(
        optimizer=optimizer_adam,
        step_size=2,
        gamma=0.1
    )
    lr_warmup_adam = get_linear_schedule_with_warmup(
        optimizer=optimizer_adam,
        num_warmup_steps=total_steps_2 / 10,
        num_training_steps=total_steps_2
    )
    lr_decay_sgd = StepLR(
        optimizer=optimizer_sgd,
        step_size=2,
        gamma=0.1
    )
    lr_warmup_sgd = get_linear_schedule_with_warmup(
        optimizer=optimizer_sgd,
        num_warmup_steps=total_steps_2 / 10,
        num_training_steps=total_steps_2
    )
    return train_dataloader, test_dataloader, lr_decay_adam, lr_warmup_adam, lr_decay_sgd, lr_warmup_sgd

def prepare_rank_data(optimizer_adam, optimizer_sgd, epochRank, idx):
    X = []
#     if not os.path.exists('./data/pairwise_x.txt'):
#         hdfs.download('./data/pairwise_x.txt', data_path+'pairwise_clean_x.txt')
    with open('./data/pairwise_x.txt', 'r') as f:
        content = f.readlines()
        for line in content:
            X.append(line)
    X = np.array(X)

    Y = []
    #     if not os.path.exists('./data/pairwise_y.txt'):
    #         hdfs.download('./data/pairwise_y.txt', data_path+'pairwise_clean_y.txt')
    with open('./data/pairwise_y.txt', 'r') as f:
        content = f.readlines()
        for line in content:
            Y.append([float(x) for x in line.split()])
    Y = np.array(Y)
    
    K = 3
    kf = KFold(K, shuffle=True, random_state=22)
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for train_idx, test_idx in kf.split(X):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)
    
    x_train, x_test, y_train, y_test = X_train[idx%K], X_test[idx%K], Y_train[idx%K], Y_test[idx%K]
    train_set = MyDataSet(x_train, y_train)
    test_set = MyDataSet(x_test, y_test)
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    train_step = len(train_dataloader)
    
    # Model Arguments
    total_steps_2 = epochRank * train_step
    lr_decay_adam = StepLR(
        optimizer=optimizer_adam,
        step_size=2,
        gamma=0.1
    )
    lr_warmup_adam = get_linear_schedule_with_warmup(
        optimizer=optimizer_adam,
        num_warmup_steps=total_steps_2 / 10,
        num_training_steps=total_steps_2
    )
    lr_decay_sgd = StepLR(
        optimizer=optimizer_sgd,
        step_size=2,
        gamma=0.1
    )
    lr_warmup_sgd = get_linear_schedule_with_warmup(
        optimizer=optimizer_sgd,
        num_warmup_steps=total_steps_2 / 10,
        num_training_steps=total_steps_2
    )
    
    return train_dataloader, test_dataloader, lr_decay_adam, lr_warmup_adam, lr_decay_sgd, lr_warmup_sgd

class BertForMT(nn.Module):
    def __init__(self, input_size, hidden_size, is_lock=True):
        super(BertForMT, self).__init__()
        self.bert = BertModel(config=BertConfig.from_pretrained(config_path))
        self.input_size = input_size
        self.hidden_size = hidden_size
#         self.fc0 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc0 = nn.Linear(self.hidden_size*2, 1)
        self.fc1 = nn.Linear(self.hidden_size, 1)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop)
        nn.init.xavier_uniform_(self.fc0.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        # 是否要冻结bert base model的参数
        if is_lock:
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def bi_concat(self, hidden):
        # input:(2, batch_size, hidden_size)
        # output:(batch_size, hidden_size*2)
        all_state1 = hidden[0, :, :]
        all_state2 = hidden[1, :, :]
        hidden = torch.cat((all_state1, all_state2), dim=1)
        return hidden

    def forward(self, input_ids):
        # forward1
        hidden = self.bert(input_ids)[0] # last_hidden_state
        hidden = hidden[:, 0] # all batches' [CLS]
        
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(hidden.unsqueeze(1))
        hidden = self.bi_concat(hidden)
        hidden = self.fc0(hidden)

        return hidden

def init_model():
    # Initializing the model
    print('1.Initializing the model...\n')
    model = BertForMT(Input_size, Hidden_size, Is_lock).to(device)
    tokenizer = BertTokenizer(vocab_file=vocab_path)
    criterion_lr = nn.MSELoss()
    criterion_ce = nn.BCELoss()
    criterion_rank = nn.MarginRankingLoss(margin=0.1)
#     criterion_rank = nn.HingeEmbeddingLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': l2},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0}
    ]
    optimizer_adam = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    optimizer_sgd = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)

    return model, tokenizer, criterion_lr, criterion_ce, criterion_rank, optimizer_adam, optimizer_sgd

class MyDataSet_lr(Dataset):
    def __init__(self, datax, datay):
        self.x = datax
        self.y = datay

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        text = self.x[idx]
        ctr = self.y[idx]
        sample = {'text':text, 'ctr':ctr}
        return sample

class MyDataSet(Dataset):
    def __init__(self, datax, datay):
        self.x = datax
        self.y = datay

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        text = self.x[idx]
        label = self.y[idx]
        sample = {
            'text':text,
            'prob_total':label
        }
        return sample

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def flat_accuracy(pred_flat, y_labels):
    y_labels = y_labels.flatten()
    return np.sum(pred_flat == y_labels)/len(y_labels)

def lr_train(train_dataloaders, test_dataloaders, tokenizer, optimizer_adam, optimizer_sgd, lr_decay_adam, lr_warmup_adam, lr_decay_sgd, lr_warmup_sgd, epochLr):
    
    for epoch in range(epochLr):
        train_loss_list, eval_loss_list = [], []
        Y_preds, Y_labels = [], []
        Yt_labels, Yt_preds = [], []
        # ========================================
        #               Training
        # ========================================
        model.train()
        print('  Perform LR Training...')
        train_dataloader = train_dataloaders[epoch]
        test_dataloader = test_dataloaders[epoch]
        if epoch == epochLr-1:
            optimizer = optimizer_sgd
            lr_warmup = lr_warmup_sgd
            lr_decay = lr_decay_sgd
        else:
            optimizer = optimizer_adam
            lr_warmup = lr_warmup_adam
            lr_decay = lr_decay_adam
        for i, batch in enumerate(train_dataloader):
#             # debug
#             if i > 0:
#                 break

            if i == 0:
                start_time = datetime.datetime.now()
            text = batch['text']
            y_label = torch.tensor(np.array(batch['ctr']), requires_grad=True)
            Y_labels.extend(y_label)
            y_label = y_label.to(device)
            input = tokenizer(text, padding='longest')
            input_ids = torch.tensor(input['input_ids']).to(device)

            optimizer.zero_grad()
            y_pred = model(input_ids=input_ids)
            y_pred = y_pred.squeeze(1)
            Y_preds.extend(y_pred)
            y_pred = y_pred.to(device)
            loss = criterion_lr(y_pred, y_label).to(device)
            train_loss_list.append(loss.item())
            
            if i % 50 == 0 and i != 0:
                end_time = datetime.datetime.now()
                interval = (end_time - start_time).seconds
                print('   【Train】Epoch [{}/{}], Step [{}/{}], Loss:{}, Time:{}s'.format(epoch + 1, epochLr,
                                                                                                i + 1,
                                                                                                len(train_dataloader),
                                                                                                sum(train_loss_list)/len(train_loss_list),
                                                                                                interval))
                start_time = datetime.datetime.now()

            loss.backward()
            optimizer.step()
            lr_warmup.step()
            lr_decay.step()

        # ========================================
        #               Validation
        # ========================================
        print('  Perform LR Validation...')
        model.eval()
        for i, batch in enumerate(test_dataloader):
#             # debug
#             if i > 1:
#                 break

            if i == 0:
                start_time2 = datetime.datetime.now()
            text = batch['text']
            y_label = torch.tensor(np.array(batch['ctr']), requires_grad=True)
            Yt_labels.extend(y_label)
            y_label = y_label.to(device)
            input = tokenizer(text, padding='longest')
            input_ids = torch.tensor(input['input_ids']).to(device)

            with torch.no_grad():
                y_pred = model(input_ids=input_ids)
                y_pred = y_pred.squeeze(1)
                Yt_preds.extend(y_pred)
                y_pred = y_pred.to(device)
                loss = criterion_lr(y_pred, y_label).to(device)
                eval_loss_list.append(loss.item())
                
            if i % 50 == 0 and i != 0:
                end_time2 = datetime.datetime.now()
                interval = (end_time2 - start_time2).seconds
                print('   【Eval】Epoch [{}/{}], Step [{}/{}], Loss:{}, Time:{}s'.format(epoch + 1, epochLr,
                                                                                               i + 1,
                                                                                               len(test_dataloader),
                                                                                               sum(eval_loss_list)/len(eval_loss_list),
                                                                                               interval))
                start_time2 = datetime.datetime.now()
        
        print('Train-NDCG:{}\n'.format(ndcg_score(np.array([Y_labels]), np.array([Y_preds]) )))
        print('Test-NDCG:{}\n'.format(ndcg_score(np.array([Yt_labels]), np.array([Yt_preds]))))
        
        print('Test-R2:{}\n'.format(r2_score(Yt_labels, Yt_preds)))
        print('Train-MSE:{}\n'.format(np.sum(train_loss_list)/len(train_loss_list)))
        print('Test-MSE:{}\n'.format(np.sum(eval_loss_list)/len(eval_loss_list)))
        
    return train_loss_list, eval_loss_list


def pairwise_train(train_dataloaders, test_dataloaders, tokenizer, optimizer_adam, optimizer_sgd, lr_decay_adam, lr_warmup_adam, lr_decay_sgd, lr_warmup_sgd, epochRank):
    
    for epoch in range(epochRank):
        train_loss_list, train_acc_list = [], []
        eval_loss_list, eval_acc_list = [], []
        Y_labels, Y_preds = [], []
        Yt_labels, Yt_preds = [], []
        # ========================================
        #               Training
        # ========================================
        model.train()
        print('  Perform Ranking Training...')
        train_dataloader = train_dataloaders[epoch]
        test_dataloader = test_dataloaders[epoch]
        if epoch <= epochRank/2:
            optimizer = optimizer_adam
            lr_warmup = lr_warmup_adam
            lr_decay = lr_decay_adam
        else:
            optimizer = optimizer_sgd
            lr_warmup = lr_warmup_sgd
            lr_decay = lr_decay_sgd
            
        for i, batch in enumerate(train_dataloader):
#             # debug
#             if i > -1:
#                 break

            # forward
            if i == 0:
                start_time = datetime.datetime.now()
            text = [xx.split("|||") for xx in batch['text']]
            batch_one = [xx[0].strip() for xx in text]
            batch_two = [xx[1].strip("\n").strip() for xx in text]
            y_label = batch['prob_total']
            y_label = torch.tensor([1.0 if float(t)>0.5 else -1.0 for t in y_label])

            Y_labels.extend(y_label)
            y_label = y_label.to(device)
            input1 = tokenizer(batch_one, padding='longest')
            input1_ids = torch.tensor(input1['input_ids']).to(device)
            input2 = tokenizer(batch_two, padding='longest')
            input2_ids = torch.tensor(input2['input_ids']).to(device)

            optimizer.zero_grad()
            y1_pred = model(input_ids=input1_ids).to(device)
            y2_pred = model(input_ids=input2_ids).to(device)
            y1_pred = y1_pred.squeeze(1)
            y2_pred = y2_pred.squeeze(1)

            # compute & recording loss & accuracy
            y_pred = []
            y_pred_norm = []
            for bi in range(len(y1_pred)):
                cur = float(y1_pred[bi]) - float(y2_pred[bi])
                y_pred.append(cur)
                y_pred_norm.append(1.0 if cur>0 else -1.0)

            Y_preds.extend(y_pred)
            loss = criterion_rank(y1_pred, y2_pred, y_label).to(device)
            accuracy = flat_accuracy(y_pred_norm, np.array(y_label.cpu()))
            train_acc_list.append(accuracy)
            train_loss_list.append(loss.item())

            if i % 50 == 0 and i != 0:
                end_time = datetime.datetime.now()
                interval = (end_time - start_time).seconds
                print('   【Train】Epoch [{}/{}], Step [{}/{}], Loss:{}, Acc:{}, Time:{}s'.format(epoch + 1, epochRank,
                                                                                                i + 1,
                                                                                                len(train_dataloader),
                                                                                                sum(train_loss_list)/len(train_loss_list), 
                                                                                                accuracy,
                                                                                                interval))
                start_time = datetime.datetime.now()

            # backward
            loss.backward()
            optimizer.step()
            lr_warmup.step()
            lr_decay.step()

        # ========================================
        #               Validation
        # ========================================
        print('  Perform Ranking Validation...')
        model.eval()
        for i, batch in enumerate(test_dataloader):
#             # debug
#             if i > 1:
#                 break
                
            if i == 0:
                start_time2 = datetime.datetime.now()
            # forward
            text = [xx.split("|||") for xx in batch['text']]
            batch_one = [xx[0].strip() for xx in text]
            batch_two = [xx[1].strip("\n").strip() for xx in text]
            y_label = batch['prob_total']
            y_label = torch.tensor([1.0 if float(t) > 0.5 else -1.0 for t in y_label])
            Yt_labels.extend(y_label)
            y_label = y_label.to(device)
            input1 = tokenizer(batch_one, padding='longest')
            input1_ids = torch.tensor(input1['input_ids']).to(device)
            input2 = tokenizer(batch_two, padding='longest')
            input2_ids = torch.tensor(input2['input_ids']).to(device)

            with torch.no_grad():
                y1_pred = model(input_ids=input1_ids).to(device)
                y2_pred = model(input_ids=input2_ids).to(device)
                y1_pred = y1_pred.squeeze(1)
                y2_pred = y2_pred.squeeze(1)

            # compute & recording loss & accuracy
            y_pred = []
            y_pred_norm = []
            for bi in range(len(y1_pred)):
                cur = float(y1_pred[bi]) - float(y2_pred[bi])
                y_pred.append(cur)
                y_pred_norm.append(1.0 if cur>0 else -1.0)

            Yt_preds.extend(y_pred)
            loss = criterion_rank(y1_pred, y2_pred, y_label).to(device)
            accuracy = flat_accuracy(y_pred_norm, np.array(y_label.cpu()))
            eval_acc_list.append(accuracy)
            eval_loss_list.append(loss.item())
            
            if i % 50 == 0 and i != 0:
                end_time2 = datetime.datetime.now()
                interval = (end_time2 - start_time2).seconds
                print('   【Eval】Epoch [{}/{}], Step [{}/{}], Loss:{}, Acc:{}, Time:{}s'.format(epoch + 1, epochRank,
                                                                                               i + 1,
                                                                                               len(test_dataloader),
                                                                                               sum(eval_loss_list)/len(eval_loss_list), 
                                                                                               accuracy,
                                                                                               interval))
                start_time2 = datetime.datetime.now()

        # calculating the metrics
        preds = np.array(Y_preds)
        labels = np.array(Y_labels)
        tpreds = np.array(Yt_preds)
        tlabels = np.array(Yt_labels)

        auc = roc_auc_score(labels, preds, average='weighted')
        auc_test = roc_auc_score(tlabels, tpreds, average='weighted')
        print('Train-AUC:{}\n'.format(auc))
        print('Test-AUC:{}\n'.format(auc_test))

        ap = average_precision_score(labels, preds, average='weighted')
        ap_test = average_precision_score(tlabels, tpreds, average='weighted')
        print('Train-Average Precision Score:{}\n'.format(ap))
        print('Test-Average Precision Score:{}\n'.format(ap_test))
        
        accuracy = np.sum(train_acc_list)/len(train_acc_list)
        accuracy_test = np.sum(eval_acc_list)/len(eval_acc_list)
        print('Train-Accuracy Score:{}\n'.format(accuracy))
        print('Test-Accuracy Score:{}\n'.format(accuracy_test))

    return train_loss_list, train_acc_list, eval_loss_list, eval_acc_list


def plot_acc_loss(loss, acc=None, is_train=None):
    fig = plt.gcf()
    fig.clf()
    
    fig = plt.figure(figsize=(16,8), dpi=100)
    ax_loss = fig.add_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    ax_acc = ax_loss.twinx()   # 共享x轴
 
    # set labels
    ax_loss.set_xlabel("steps")
    ax_loss.set_ylabel("loss")
    ax_acc.set_ylabel("accuracy")
 
    # plot curves
    ax_loss.plot(range(len(loss)), loss, label="loss", color='b', linestyle='-')
    if acc != None:
        ax_acc.plot(range(len(acc)), acc, label="accuracy", color='r', linestyle='-')
 
    # set the range of x axis of host and y axis of par1
    if is_train:
        ax_loss.set_xlim([0, 2500])
    else:
        ax_loss.set_xlim([0, 900])
    ax_acc.set_ylim([-0.1, 1.1])
    
    plt.show()


if __name__ == '__main__':
    
    setup_seed(22)
    model, tokenizer, criterion_lr, criterion_ce, criterion_rank, optimizer_adam, optimizer_sgd = init_model()

    # Pairwise-rank Learning
    epochRank = 12
    train_dataloaders, test_dataloaders = [], []
    print('2.Preparing the Ranking data...\n')
    for i in range(epochRank):
        train_dataloader, test_dataloader, lr_decay_adam, lr_warmup_adam, lr_decay_sgd, lr_warmup_sgd = prepare_rank_data(optimizer_adam, optimizer_sgd, epochRank, i)
        train_dataloaders.append(train_dataloader)
        test_dataloaders.append(test_dataloader)
    train_loss_list, train_acc_list, eval_loss_list, eval_acc_list = pairwise_train(train_dataloaders, test_dataloaders, tokenizer, optimizer_adam, optimizer_sgd, lr_decay_adam, lr_warmup_adam, lr_decay_sgd, lr_warmup_sgd, epochRank)
    plot_acc_loss(train_loss_list, train_acc_list, is_train=True)
    plot_acc_loss(eval_loss_list, eval_acc_list, is_train=False)
    
    # Linear Regression
    epochLr = 12
    train_dataloaders, test_dataloaders = [], []
    print('3.Preparing the LR data...\n')
    for i in range(epochLr):
        train_dataloader, test_dataloader, lr_decay_adam, lr_warmup_adam, lr_decay_sgd, lr_warmup_sgd = prepare_lr_data(optimizer_adam, optimizer_sgd, epochLr, i)
        train_dataloaders.append(train_dataloader)
        test_dataloaders.append(test_dataloader)
    train_loss_list, eval_loss_list = lr_train(train_dataloaders, test_dataloaders, tokenizer, optimizer_adam, optimizer_sgd, lr_decay_adam, lr_warmup_adam, lr_decay_sgd, lr_warmup_sgd, epochLr)
    plot_acc_loss(train_loss_list, is_train=True)
    plot_acc_loss(eval_loss_list, is_train=False)

