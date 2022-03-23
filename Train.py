import numpy as np 
# np.save('data.npy', train_data) # save
from tkinter import filedialog

path2Data = filedialog.askopenfilename(initialdir = '.', initialfile='train_data.npy')  #открыть обучающие данные из файла numpy
path2Labels = filedialog.askopenfilename(initialdir = '.', initialfile='Main_data.csv') #открыть прецедентные данные (labels) 

train_data = np.load(path2Data)

import pandas as pd
train_labels = pd.read_csv(path2Labels, header = None)[1]   #Выбрать столбец, на который тренируем

import tensorflow as tf
#from tf.keras import tf.keras.layers
#from tf.keras import tf.keras.losses
#from tf.keras import tf.keras.metrics
#from tf.keras import tf.keras.optimizers

#Архитектура сети, с которой и нужно экспериментировать
model = tf.keras.models.Sequential()    
model.add(tf.keras.layers.Dense(500, activation='relu', input_shape=(75000,)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.0001),
              loss= tf.keras.losses.MeanSquaredError())

x_val = train_data[1000:]
partial_x_train = train_data[:1000]
y_val = train_labels[1000:]
partial_y_train = train_labels[:1000]

#Запуск обучения
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size=200,
                    validation_data=(x_val, y_val))