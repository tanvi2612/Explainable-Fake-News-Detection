from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split 
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('../model/out', output_attentions=True)
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

tokenizer = BertTokenizer.from_pretrained('../model/out')

df = pd.read_csv('../data/final1.csv')
df = df.dropna()
df = df[df['Len']<360]

x = list(df.Text)
y = df.Fake 
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
df1 = pd.DataFrame({'Text':x_test, 'Fake':y_test})
df1.to_csv('../data/test1.csv')

x = tokenizer(
    text=x_train,
    add_special_tokens=True,
    truncation=True,
    padding=True,
    return_tensors='pt',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True
)

y = torch.tensor(y_train)
train_data = TensorDataset(x['input_ids'], y)
train_dataloader = DataLoader(train_data, batch_size=4)

print(f"Length DataLoader = {len(train_dataloader)}")

for e in range(40):
    print(f"Epoch {e+1}")
    t0_epoch, t0_batch = time.time(), time.time()
    total_loss, batch_loss, batch_counts = 0, 0, 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch_counts +=1
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        logits = model(b_input_ids).logits
        loss = torch.nn.functional.cross_entropy(logits, b_labels)
        batch_loss += loss.item()
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step % 200 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            time_elapsed = time.time() - t0_batch
            print(f"Step : {step} Loss : {batch_loss / batch_counts} Time : {time_elapsed}")
            batch_loss, batch_counts = 0, 0
            t0_batch = time.time()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch Loss : {avg_train_loss}")
    if (e+1)%5==0:
        PATH = '../model/pretrained'+str(e+1)+'.pt'
        torch.save(model, PATH)





