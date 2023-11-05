import numpy as np
import tensorflow as tf

EPOCH = 20
a = np.load('1.npy')
b = np.load('2.npy')
c = np.load('2.npy')

x_train =  np.concatenate((a , b , c)) # lista macierzy o rozmiarze (21, 3)
y_train = np.array([0 for _ in a] + [1 for _ in b] + [2 for _ in c]) # liczba który symbol


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

model.save('test.model')