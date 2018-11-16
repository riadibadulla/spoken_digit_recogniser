#!/usr/bin/env python
#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn
import numpy
numpy.set_printoptions(threshold=numpy.nan)
import speech_data

learning_rate = 0.0001
training_iters = 1 # steps
batch_size = 32

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

batch = word_batch = speech_data.mfcc_batch_generator(batch_size)
testBatch = speech_data.validationSet()
#testX, testY = testBatch



# Network building
net = tflearn.input_data([None, width, height])
# net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, tensorboard_verbose=3)
model.load("myModel1")


while training_iters > 0: #training_iters
  training_iters -= 1
  X, Y = next(batch)
  trainX, trainY = X, Y
  testX, testY = X, Y
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
  weights = model.get_weights(net.W)
  
  #print(weights)
  #_y=model.predict(X)
model.save("myModel1")
for i in range (10):
  print(testY[i])
print(model.predict(testBatch[0]))
#print (z)
#print (Y)
