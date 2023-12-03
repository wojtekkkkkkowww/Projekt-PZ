import numpy as np
import tensorflow as tf
import os

DATA_DIR = 'data/model1/sample1'
EPOCH = 200

test = []
for i in range(12):
    test.append(np.load(os.path.join(DATA_DIR, f"{i}.npy")))


x_train =  np.concatenate(test) 
y_train = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(test)])

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

model.save('models/model1.keras')
tf.saved_model.save(model, 'to_lite_data/model1')
