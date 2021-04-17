import os
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from typing import List
import torch

def run(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForSequenceClassification.from_pretrained('finetuned3')


    inputs = tokenizer(
        text=text,
        return_tensors='tf',
    )
    #print(len(inputs['input_ids'][0]))
    out1 = []
    if len(inputs['input_ids'][0]) < 512:
        outputs = model(inputs, output_attentions=True)
        for t in range(12):
            attn = outputs.attentions[t][0]
            lab = outputs.logits.numpy()[0][0]
            b = np.sum(attn, axis=0)
            out = np.sum(b, axis=0)
            inp = inputs['input_ids'][0]
            high = 0
            for i in range(1, len(out)-10):
                if out[i]+out[i+1]+out[i+2]+out[i+3]+out[i+4]+out[i+5]+out[i+6]+out[i+7]+out[i+8]+out[i+9] > high:
                    high = out[i]+out[i+1]+out[i+2]+out[i+3]+out[i+4]+out[i+6]+out[i+7]+out[i+8]+out[i+9]
                    maxi = i
            out1.append(str(tokenizer.decode(inp[maxi]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+1]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+2]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+3]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+4]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+5]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+6]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+7]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+8]).replace(' ','') + ' ' + tokenizer.decode(inp[maxi+9]).replace(' ','')))

    return (out1, lab)

def run1(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForSequenceClassification.from_pretrained('finetuned3')


    inputs = tokenizer(
        text=text,
        return_tensors='tf',
    )
    ind = np.where(inputs['input_ids'].numpy()[0]==119)[0]
    out1 = []
    if len(inputs['input_ids'][0]) < 512:
        outputs = model(inputs, output_attentions=True)
        lab = outputs.logits.numpy()[0][0]
        if lab > 0.65:
            lab = False
        else:
            lab = True
        if not lab:
            for t in range(12):
                attn = outputs.attentions[t][0]
                b1 = np.sum(attn, axis=0)
                b = np.sum(b1, axis=0)
                inp = inputs['input_ids'][0]
                maxv = 0
                maxi = -1
                for i in range(len(ind)-1):
                    v = sum(b[ind[i]+1:ind[i+1]+1])/(ind[i+1]-ind[i])
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
                    print(out)
                    out1.append(out)
                
            maxc = 0
            fin = ''
            for sent in out1:
                if out1.count(sent) > maxc:
                    maxc = out1.count(sent)
                    fin = sent    
        else:
            fin = 'N/A'

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
        lab = torch.nn.Softmax(dim=1)(outputs.logits).detach().numpy()[0]
        if 0.9*lab[1] > lab[0]:
            string = True
        else:
            string = False
        if not string:
            for t in range(12):
                attn = outputs.attentions[t][0].detach().numpy()   
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
                    for i in range(ind[maxi]+1,ind[maxi+1]+1):
                        out += str(tokenizer.decode(inp[i]).replace(' ',''))+' '
                    out = out.replace(' ##', '')            
                    out1.append(out)
                
            maxc = 0
            fin = ''
            for sent in out1:
                if out1.count(sent) > maxc:
                    maxc = out1.count(sent)
                    fin = sent   
        else:
            fin = 'N/A'

    return (fin, string)

app = FastAPI(title="Detectr", version="0.0.1", docs_url="/api")
templates = Jinja2Templates(directory="templates")

PWD = os.getcwd()

# Models
class PromptIn(BaseModel):
    article_text: str

class PromptOut(BaseModel):
    label: bool
    exp: str

# Helper function
def process_text(text):
    output = run2(text)
    out = {
        "label": output[1],
        "exp": output[0]
    }
    return out

# Routes
@app.post("/api/process", response_model=PromptOut, status_code=200)
def process(prompt_in: PromptIn):
    print(prompt_in)
    response = process_text(prompt_in.article_text)  
    return PromptOut(**response)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def results(request: Request):
    data = await request.form()
    result = process_text(data['prompt'])
    return templates.TemplateResponse("results.html", {"request": request, "result": result})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8888, reload=True)
