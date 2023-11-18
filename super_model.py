import numpy as np
import tensorflow as tf
import os

DATA_DIR = 'data/model1/sample1'
EPOCH = 200

test = []
for i in range(12):
    test.append(np.load(os.path.join(DATA_DIR, f"{i}.npy")))

model1 = tf.keras.models.load_model(f"models/model1.keras")
model2 = tf.keras.models.load_model(f"models/model2.keras")
model2 = tf.keras.models.load_model(f"models/model3.keras")



y_train = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(test)])
x_train = []
for sign in test:
    p1 = model1.predict(sign)
    p2 = model1.predict(sign)
    p3 = model1.predict(sign)
    x_train.append([p1,p2,p3])    
x_train = np.array(x_train)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(12, 3)),
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
