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
model3 = tf.keras.models.load_model(f"models/model3.keras")

p_model1 = tf.keras.Sequential([model1, 
                                         tf.keras.layers.Softmax()])
p_model2 = tf.keras.Sequential([model2, 
                                         tf.keras.layers.Softmax()])
p_model3 = tf.keras.Sequential([model3, 
                                         tf.keras.layers.Softmax()])

y_train = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(test)])
x_train = []
test =  np.concatenate(test)
print(len(test))
for i,sign in enumerate(test):
    print(i)
    p1 = p_model1.predict(np.array([sign]))
    p2 = p_model2.predict(np.array([sign]))
    p3 = p_model3.predict(np.array([sign]))
    x_train.append(np.concatenate([p1,p2,p3]))    
x_train = np.array(x_train)
print(x_train.shape)

np.save(f"data/supermodel/x_train.npy",x_train)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(12, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(12) 
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=EPOCH)

model.save('models/supermodel.keras')
