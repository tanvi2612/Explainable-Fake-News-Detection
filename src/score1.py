from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import transformers
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
for it in range(1):
    print(f"Epoch No. {it}")
    string = '../model/torch.pt'
    model = torch.load(string)
    model.to(device)

    df = pd.read_csv('../data/test1.csv')

    x = list(df.Text)[:300]
    y=df.Fake[:300]

    inputs = tokenizer(
        text=x,
        max_length = 512,
        truncation=True,
        padding=True,
        add_special_tokens=True,
        return_tensors='pt',
    )
    y = torch.tensor(y)

    test_data = TensorDataset(inputs['input_ids'], y)
    test_dataloader = DataLoader(test_data, batch_size=1)


    for cutoff in [0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 1.1, 1.125, 1.25]:
        print(f"Cutoff = {cutoff}")
        y_out = []

        for step, batch in enumerate(test_dataloader):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            outputs = model(b_input_ids)
            out = torch.nn.Softmax(dim=1)(outputs.logits).cpu().detach().numpy()[0]

            if cutoff*out[1] > out[0]:
                y_out.append(1)
            else:
                y_out.append(0)

        print(classification_report(y_out, y))


