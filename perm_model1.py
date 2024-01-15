import numpy as np
import tensorflow as tf
import os

DATA_DIR = 'data/model1/sample1'
EPOCH = 100

key = np.arange(21*3)
np.random.shuffle(key)
np.save("key.npy", key)

test = []
for i in range(12):
    test.append(np.load(os.path.join(DATA_DIR, f"{i}.npy")))


x_train =  np.concatenate(test) 
y_train = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(test)])
for x in x_train:
    x = x.flatten()
    x = x[key]
    x = x.reshape(21, 3)
 

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(12) 
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=EPOCH)

model.save('models/perm_model1.keras')
