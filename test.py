import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import TextVectorization
df=pd.read_csv("train.csv")
X=df['comment_text']
model=tf.keras.models.load_model('toxicity3.h5')
MAX_FEATURES=200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,output_sequence_length=1800,output_mode='int')
vectorizer.adapt(X.values)
vectorizer.get_vocabulary()
input='hi how are you'
input2=vectorizer(input)
res=model.predict(np.expand_dims(input2,0))
print("comment: ",input)
print('toxic, severe toxic, obscene, threat, insult, identity hate')
print(res)
print('loaded')
