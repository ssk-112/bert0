from transformers import BertTokenizer,BertModel,BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import logging
logging.set_verbosity_error()
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import argparse

df = pd.read_csv('./s_cleaned_file.csv')
df.info()
df['label'].value_counts()
#调整数据
df['label'] = df['label'].replace(4, 1)

x = list(df['text'])
y = list(df['label'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2)
tokenizer = BertTokenizer.from_pretrained(r'D:\CS1501\bert0\bert')
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)

#可以这样查看词典
vocab = tokenizer.vocab
print(vocab['hello'])

print(train_encoding.keys())

# 数据集读取, 继承torch的Dataset类，方便后面用DataLoader封装数据集
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    #这里的idx是为了让后面的DataLoader成批处理成迭代器，按idx映射到对应数据
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item
    
    #数据集长度。通过len(这个实例对象)，可以查看长度
    def __len__(self):
        return len(self.labels)
#将数据集包装成torch的Dataset形式
train_dataset = NewsDataset(train_encoding, y_train)
test_dataset = NewsDataset(test_encoding, y_test)
# 单个读取到批量读取
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

#可以看看长啥样
batch = next(iter(train_loader))  
print(batch)
print(batch['input_ids'].shape)

class my_bert_model(nn.Module):
    def __init__(self, freeze_bert=False, hidden_size=768):
        super().__init__()
        config = BertConfig.from_pretrained(r'D:\CS1501\bert0\bert')
        config.update({'output_hidden_states':True})
        self.bert = BertModel.from_pretrained(r'D:\CS1501\bert0\bert',config=config)
        self.fc = nn.Linear(hidden_size*4, 2)
        
        #是否冻结bert，不让其参数更新
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        all_hidden_states = torch.stack(outputs[2])  #因为输出的是所有层的输出，是元组保存的，所以转成矩阵
        concat_last_4layers = torch.cat((all_hidden_states[-1],   #取最后4层的输出
                                         all_hidden_states[-2], 
                                         all_hidden_states[-3], 
                                         all_hidden_states[-4]), dim=-1)
        
        cls_concat = concat_last_4layers[:,0,:]   #取 [CLS] 这个token对应的经过最后4层concat后的输出
        result = self.fc(cls_concat)
        
        return result
     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device,'能用')

model = my_bert_model().to(device)
criterion = nn.CrossEntropyLoss().to(device)
train_labels = []
train_outputs = []
# 优化方法
#过滤掉被冻结的参数，反向传播需要更新的参数
optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optim, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def plot(x, colors):
    # Choosing color palette
    # https://seaborn.pydata.org/generated/seaborn.color_palette.html
    palette = np.array(sns.color_palette("pastel", 2))
    # pastel, husl, and so on

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    # Add the labels for each digit.
    txts = []
    for i in range(2):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    plt.savefig('./digits_tsne-pastel.png', dpi=120)
    return f, ax, txts

# 训练函数
def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    print('总数',total_iter)
    
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        print(labels)
        label = [t.tolist() for t in labels]
        l_length = len(label)
        for i in range(l_length):
            train_labels.append(label[i])
        outputs = model(input_ids, attention_mask=attention_mask)
        output = [t.tolist() for t in outputs]
        output_length = len(output)
        for i in range(output_length):
            train_outputs.append(output[i])
        loss = criterion(outputs, labels)                  
        total_train_loss += loss.item()
           
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   #梯度裁剪，防止梯度爆炸
        
        # 参数更新
        optim.step()
        scheduler.step()

        iter_num += 1
        print('训练数',iter_num)
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
        
    print("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(train_loader)))

# 精度计算
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    total_tp, total_fp, total_fn = 0, 0, 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)
            logits = outputs

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            total_tp += np.sum(np.argmax(logits, axis=1) == label_ids)
            fp = np.sum(np.argmax(logits, axis=1) != label_ids) - np.sum(label_ids == 0)
            total_fp += abs(fp)
            fn = np.sum(label_ids == 0) - np.sum(np.argmax(logits, axis=1) == label_ids)
            total_fn += abs(fn)
            
    print('总acc',total_eval_accuracy)
    print(total_tp)
    print(total_fp)
    print(total_fn)
    
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * precision * recall / (precision + recall)

    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Precision: %.4f" % (precision))
    print("Recall: %.4f" % (recall))
    print("F1 score: %.4f" % (f1))
    print("Average testing loss: %.4f"%(total_eval_loss/len(test_dataloader)))
    print("-------------------------------")

n_epochs = 1
for epoch in range(n_epochs):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()
    print(train_outputs)
    print(train_labels)
    array1 = np.array(train_outputs)
    array2 = np.array(train_labels)
    out = {'data': array1,'target': array2}
    out = argparse.Namespace(**out)
    digits = out
    print(digits)
    print(digits.data.shape)
# There are 10 classes (0 to 9) with alomst 180 images in each class 
# The images are 8x8 and hence 64 pixels(dimensions)

# Place the arrays of data of each digit on top of each other and store in X
    X = np.vstack([digits.data[digits.target==i] for i in range(2)])
# Place the arrays of data of each target digit by the side of each other continuosly and store in Y
    Y = np.hstack([digits.target[digits.target==i] for i in range(2)])    
# Implementing the TSNE Function - ah Scikit learn makes it so easy!
    digits_final = TSNE(perplexity=30).fit_transform(X) 
# Play around with varying the parameters like perplexity, random_state to get different plots
    plot(digits_final, Y)
