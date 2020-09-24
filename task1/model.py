import datetime
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, BertConfig, BertForPreTraining

# hyper-parameter
K = 3
batch_size = 32
epoch = 3
l2 = 0
learning_rate = 2e-5
Input_size = 768
Hidden_size = 768
model_path = '/home/xiaosuliu/chinese_L-12_H-768_A-12/'
data_path = '/home/xiaosuliu/codebase/1-tomcat'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = os.path.join(model_path, 'config.json')
checkpoint_path = os.path.join(model_path, 'model.ckpt.index')
vocab_path = os.path.join(model_path, 'vocab.txt')

class BertForClassify(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BertForClassify, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bert = BertModel(config=BertConfig.from_pretrained(config_path))
        self.fc = nn.Linear(self.hidden_size*2, 1)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True,
                            bidirectional=True)
        nn.init.xavier_uniform_(self.fc.weight)

    def bi_concat(self, hidden):
        # input:(2, batch_size, hidden_size)
        # output:(batch_size, hidden_size*2)
        all_state1 = hidden[0, :, :]
        all_state2 = hidden[1, :, :]
        hidden = torch.cat((all_state1, all_state2), dim=1)
        return hidden

    def forward(self, input_ids):
        hidden = self.bert(input_ids)[0]
        hidden = hidden[:, 0] # all sample in batch's [CLS]
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(hidden.unsqueeze(1))
        hidden = self.bi_concat(hidden)
        hidden = self.fc(hidden)
        return hidden

def init_model():
    print('Initializing the model...')
    model = BertForClassify(Input_size, Hidden_size)
    tokenizer = BertTokenizer(vocab_file=vocab_path)
    criterion_ce = nn.BCEWithLogitsLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': l2},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0}
    ]
    optimizer_adam = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    optimizer_sgd = torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate)

    return model, tokenizer, criterion_ce, optimizer_adam, optimizer_sgd

class MyDataSet(Dataset):
    def __init__(self, datax, datay):
        super(MyDataSet, self).__init__()
        self.x = datax
        self.y = datay

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

def prepare_data(idx):

    df = pd.read_csv(os.path.join(data_path, 'allData.csv'))
    X = np.array(df['func'])
    Y = np.array(df['comm'])

    # K-Fold cross validation
    kf = KFold(n_splits=K)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for train_idx, test_idx in kf.split(X):
        x_train, y_train = np.array(X)[train_idx], np.array(Y)[train_idx]
        x_test, y_test = np.array(X)[test_idx], np.array(Y)[test_idx]
        X_train.append(x_train)
        Y_train.append(y_train)
        X_test.append(x_test)
        Y_test.append(y_test)

    x_train, y_train, x_test, y_test = X_train[idx%K], Y_train[idx%K], X_test[idx%K], Y_test[idx%K]
    train_set = MyDataSet(x_train, y_train)
    test_set = MyDataSet(x_test, y_test)
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_dataloader, test_dataloader

def flat_accuracy(pred_flat, y_labels):
    y_labels = y_labels.flatten()
    return np.sum(pred_flat == y_labels)/len(y_labels)

def classify_train(model, train_dataloaders, test_dataloaders, tokenizer, optimizer_adam, optimizer_sgd):

    for e in range(epoch):

        train_loss_list, train_acc_list = [], []
        eval_loss_list, eval_acc_list = [], []
        Y_labels, Y_preds = [], []
        Yt_labels, Yt_preds = [], []

        # ========================================
        #               Training
        # ========================================
        model.train()
        print('Perform Classifying Training...')
        train_dataloader = train_dataloaders[e]
        test_dataloader = test_dataloaders[e]
        if e <= epoch/2:
            optimizer = optimizer_adam
        else:
            optimizer = optimizer_sgd
        for i, batch in enumerate(train_dataloader):
            if i == 0:
                start_time = datetime.datetime.now()
            y_label = torch.tensor([t for t in batch[1]]).to(device)
            Y_labels.extend(y_label)
            input = tokenizer(batch[0], padding='longest')
            input_ids = torch.tensor(input['input_ids']).to(device)

            optimizer.zero_grad()
            y_pred = model(input_ids=input_ids).to(device)
            y_pred = y_pred.squeeze(1)
            y_pred_norm = [1.0 if x > 0.5 else 0.0 for x in y_pred]

            # compute & recording loss & accuracy
            Y_preds.extend(y_pred)
            loss = criterion_ce(y_pred, y_label).to(device)
            accuracy = flat_accuracy(y_pred_norm, np.array(y_label.cpu()))
            train_acc_list.append(accuracy)
            train_loss_list.append(loss.item())

            if i % 50 == 0 and i != 0:
                end_time = datetime.datetime.now()
                interval = (end_time - start_time).seconds
                print('   【Train】Epoch [{}/{}], Step [{}/{}], Loss:{}, Acc:{}, Time:{}s'.format(e + 1, epoch,
                                                                                                i + 1,
                                                                                                len(train_dataloader),
                                                                                                sum(train_loss_list)/len(train_loss_list),
                                                                                                accuracy,
                                                                                                interval))
                start_time = datetime.datetime.now()

            # backward
            loss.backward()
            optimizer.step()

        # ========================================
        #               Validation
        # ========================================
        print('Perform Classifying Validation...')
        model.eval()
        for i, batch in enumerate(test_dataloader):
            if i == 0:
                start_time2 = datetime.datetime.now()
            y_label = torch.tensor([t for t in batch[1]]).to(device)
            Y_labels.extend(y_label)
            input = tokenizer(batch[0], padding='longest')
            input_ids = torch.tensor(input['input_ids']).to(device)

            optimizer.zero_grad()
            y_pred = model(input_ids=input_ids).to(device)
            y_pred = y_pred.squeeze(1)
            y_pred_norm = [1.0 if x > 0.5 else 0.0 for x in y_pred]

            # compute & recording loss & accuracy
            Y_preds.extend(y_pred)
            loss = criterion_ce(y_pred, y_label).to(device)
            accuracy = flat_accuracy(y_pred_norm, np.array(y_label.cpu()))
            eval_acc_list.append(accuracy)
            eval_loss_list.append(loss.item())

            if i % 50 == 0 and i != 0:
                end_time2 = datetime.datetime.now()
                interval = (end_time2 - start_time2).seconds
                print('   【Eval】Epoch [{}/{}], Step [{}/{}], Loss:{}, Acc:{}, Time:{}s'.format(e + 1, epoch,
                                                                                               i + 1,
                                                                                               len(test_dataloader),
                                                                                               sum(
                                                                                                   eval_loss_list) / len(
                                                                                                   eval_loss_list),
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

        accuracy = np.sum(train_acc_list) / len(train_acc_list)
        accuracy_test = np.sum(eval_acc_list) / len(eval_acc_list)
        print('Train-Accuracy Score:{}\n'.format(accuracy))
        print('Test-Accuracy Score:{}\n'.format(accuracy_test))

    return train_loss_list, train_acc_list, eval_loss_list, eval_acc_list


def plot_acc_loss(loss, acc=None, is_train=None):
    fig = plt.gcf()
    fig.clf()

    fig = plt.figure(figsize=(16, 8), dpi=100)
    ax_loss = fig.add_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    ax_acc = ax_loss.twinx()  # 共享x轴

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
    torch.set_default_tensor_type(torch.DoubleTensor)
    model, tokenizer, criterion_ce, optimizer_adam, optimizer_sgd = init_model()
    train_dataloaders, test_dataloaders = [], []
    print('Preparing the data...')
    for i in range(epoch):
        train_loader, test_loader = prepare_data(i)
        train_dataloaders.append(train_loader)
        test_dataloaders.append(test_loader)
    train_loss_list, train_acc_list, eval_loss_list, eval_acc_list = classify_train(model, train_dataloaders, test_dataloaders,
                                                                                    tokenizer, optimizer_adam,
                                                                                    optimizer_sgd)
    plot_acc_loss(train_loss_list, train_acc_list, is_train=True)
    plot_acc_loss(eval_loss_list, eval_acc_list, is_train=False)