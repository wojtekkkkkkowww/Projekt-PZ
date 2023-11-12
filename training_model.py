import numpy as np
import tensorflow as tf
import sys

DATA_DIR = 'model1_data'
EPOCH = 100

model = tf.keras.models.load_model(sys.argv[1])
test = []
for i in range(12):
    test.append(np.load( f"{i}.npy"))


x_train =  np.concatenate(test) 
y_train = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(test)])
model.fit(x_train, y_train, epochs=EPOCH)

model.save(sys.argv[1])