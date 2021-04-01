import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
import torch

def run(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForSequenceClassification.from_pretrained('finetuned3')


    inputs = tokenizer(
        text=text,
        return_tensors='tf',
    )
    ind = np.where(inputs['input_ids'].numpy()[0]==119)[0]
    print(ind)
    out1 = []
    if len(inputs['input_ids'][0]) < 512:
        outputs = model(inputs, output_attentions=True)
        for t in range(12):
            attn = outputs.attentions[t][0]
            lab = outputs.logits.numpy()[0][0]
            b1 = np.sum(attn, axis=0)
            b = np.sum(b1, axis=0)
            inp = inputs['input_ids'][0]
            maxv = 0
            maxi = -1
            for i in range(len(ind)-1):
                v = sum(b[ind[i]:ind[i+1]+1])/(ind[i+1]-ind[i])
                if v > maxv:
                    maxv = v
                    maxi = i
            
            out = ''
            if maxi == -1:
                for i in range(0,ind[0]):
                    out += str(tokenizer.decode(inp[i]).replace(' ',''))+' '
                out = out.replace(' ##', '')
            else:
                for i in range(ind[maxi],ind[maxi+1]+1):
                    out += str(tokenizer.decode(inp[i]).replace(' ',''))+' '
                out = out.replace(' ##', '')            
                out1.append(out)
            
            maxc = 0
            fin = ''
            for sent in out1:
                if out1.count(sent) > maxc:
                    maxc = out1.count(sent)
                    fin = sent

    return (fin, lab)

def run2(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = torch.load('torch.pt', map_location=torch.device('cpu'))


    inputs = tokenizer(
        text=text,
        return_tensors='pt',
    )
    ind = np.where(inputs['input_ids'].numpy()[0]==119)[0]
    out1 = []
    if len(inputs['input_ids'][0]) < 512:
        outputs = model(inputs['input_ids'])
        for t in range(12):
            attn = outputs.attentions[t][0].detach().numpy()
            lab = outputs.logits.detach().numpy()[0][0]
            b1 = np.sum(attn, axis=0)
            b = np.sum(b1, axis=0)
            inp = inputs['input_ids'][0]
            maxv = 0
            maxi = -1
            for i in range(len(ind)-1):
                v = sum(b[ind[i]:ind[i+1]+1])/(ind[i+1]-ind[i])
                if v > maxv:
                    maxv = v
                    maxi = i
            
            out = ''
            if maxi == -1:
                for i in range(0,ind[0]):
                    out += str(tokenizer.decode(inp[i]).replace(' ',''))+' '
                out = out.replace(' ##', '')
            else:
                for i in range(ind[maxi],ind[maxi+1]+1):
                    out += str(tokenizer.decode(inp[i]).replace(' ',''))+' '
                out = out.replace(' ##', '')            
                out1.append(out)
            
        maxc = 0
        fin = ''
        for sent in out1:
            if out1.count(sent) > maxc:
                maxc = out1.count(sent)
                fin = sent    

    return (fin, lab)


text = 'Rhea and Showik Chakraborty are currently being grilled by the Narcotics Control Bureau and the latest revelation by the siblings now involves several Bollywood celebrities. According to Times Now, during the interrogation, Rhea gave names of several actors, directors and producers who are allegedly related to a drugs cartel. The actress has also reportedly mention recent Bollywood parties and how drugs were smuggled. According to the channel, in the next 10-15 days the NCB is expected to start issuing summons to the celebs names, after proper verification.'
print(run2(text))