import pickle
import json
import random

# NLP stuff
import nltk
nltk.download('punkt')
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer


# TensorFlow stuff
import numpy as np
import tflearn
import tensorflow as tf
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
stemmer = LancasterStemmer()
# from nltk.stem.snowball import FrenchStemmer
# stemmer = FrenchStemmer()

# load json file
with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignored_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignored_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words ]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

pickle.dump({
    'words': words,
    'classes': classes,
    'train_x': train_x,
    'train_y': train_y
    }, open('training_data', 'wb'))
