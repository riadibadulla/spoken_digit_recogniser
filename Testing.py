#!/usr/bin/env python
#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tflearn
import speech_data
import pyaudio
import wave
import numpy as np



learning_rate = 0.0001
training_iters = 1 # steps
batch_size = 64
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 4096
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "new.wav"
recognisedNumber = 0
probabilityOfRecognisedDegit = 0


width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

batch = word_batch = speech_data.mfcc_batch_generator(batch_size)
X, Y = next(batch)

trainX, trainY = X, Y
testX, testY = X, Y #overfit for now

# Data preprocessing
# Sequence padding
# trainX = pad_sequences(trainX, maxlen=100, value=0.)
# testX = pad_sequences(testX, maxlen=100, value=0.)
# # Converting labels to binary vectors
# trainY = to_categorical(trainY, nb_classes=2)
# testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, width, height])
# net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.load("myModel1")

def record():
    global recognisedNumber, probabilityOfRecognisedDegit
    audio = pyaudio.PyAudio()

    
    # Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")

     
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open("0_new.wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()



 
    z = speech_data.loadPredict("","0_new.wav")
    wav, lab = z
    predicted = model.predict(z[0])
    print(predicted[0])
    print("The number is ... ",np.argmax(predicted[0]))
    recognisedNumber = np.argmax(predicted[0])
    probabilityOfRecognisedDegit = predicted[0][recognisedNumber]