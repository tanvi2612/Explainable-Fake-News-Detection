import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split 
from sklearn.utils import class_weight


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=1)

optimizer = tf.keras.optimizers.Adam(lr = 2e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric = tf.keras.metrics.BinaryAccuracy('accuracy')
model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metric
)

df = pd.read_csv('../data/final1.csv')
df = df.dropna()
df = df[df['Len']<360]

x = list(df.Text)
y = df.Fake 
y = np.array(y)

'''
y1 = []
for i in y:
    if i == 0:
        y1.append([1,0])
    else:
        y1.append([0,1])
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
df1 = pd.DataFrame({'Text':x_test, 'Fake':y_test})
df1.to_csv('../data/test.csv')

x = tokenizer(
    text=x_train,
    add_special_tokens=True,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True
)

class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                         classes=np.unique(y),
                                                         y=y)))

y=np.array(y_train)

history = model.fit(
    x={'input_ids': x['input_ids']},
    y=y,
    class_weight=class_weights,
    validation_split=0.1,
    batch_size=32,
    epochs=1
)

model.save_pretrained("../model/finetuned1/")

history = model.fit(
    x={'input_ids': x['input_ids']},
    y=y,
    validation_split=0.1,
    batch_size=32,
    epochs=1
)

model.save_pretrained("../model/finetuned2/")

history = model.fit(
    x={'input_ids': x['input_ids']},
    y=y,
    validation_split=0.1,
    batch_size=32,
    epochs=1
)

model.save_pretrained("../model/finetuned3/")

history = model.fit(
    x={'input_ids': x['input_ids']},
    y=y,
    validation_split=0.1,
    batch_size=32,
    epochs=1
)

model.save_pretrained("../model/finetuned4/")
