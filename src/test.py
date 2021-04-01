import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('../model/finetuned3')

df = pd.read_csv('../data/test.csv')

x = list(df.Text)
y=df.Fake

inp = 66
print(y[inp])
text = x[inp]
print(x[inp])

inputs = tokenizer(
    text=text,
    return_tensors='tf',
)
print(len(inputs['input_ids'][0]))
if len(inputs['input_ids'][0]) < 512 and y[inp]==0:
    outputs = model(inputs, output_attentions=True)
    for t in range(12):
        attn = outputs.attentions[t][0]
        b = np.sum(attn, axis=0)
        out = np.sum(b, axis=0)
        inp = inputs['input_ids'][0]
        high = 0
        for i in range(1, len(out)-10):
            if out[i]+out[i+1]+out[i+2]+out[i+3]+out[i+4]+out[i+5]+out[i+6]+out[i+7]+out[i+8]+out[i+9] > high:
                high = out[i]+out[i+1]+out[i+2]+out[i+3]+out[i+4]+out[i+6]+out[i+7]+out[i+8]+out[i+9]
                maxi = i
        print(tokenizer.decode(inp[maxi]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+1]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+2]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+3]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+4]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+5]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+6]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+7]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+8]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+9]).replace(' ',''))


