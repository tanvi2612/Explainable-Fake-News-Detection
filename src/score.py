from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('../model/finetuned4')

df = pd.read_csv('../data/test.csv')

x = list(df.Text)[:300]
y=df.Fake[:300]

for cutoff in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
    print(f"Cutoff : {cutoff}")
    y_out = []

    for inp in x:
        inputs = tokenizer(
            text=inp,
            max_length = 512,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors='tf',
        )
        outputs = model(inputs)
        out = outputs.logits.numpy()[0][0]
        if out > cutoff:
            y_out.append(1)
        else:
            y_out.append(0)

    print(classification_report(y_out, y))


