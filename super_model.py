import numpy as np
import tensorflow as tf
import os

DATA_DIR = 'data/model1/sample1/'
EPOCH = 20

test = []
for i in range(12):
    test.append(np.load(os.path.join(DATA_DIR, f"{i}.npy")))


i1 = tf.lite.Interpreter(model_path='./lite/model1.tflite')
i2 = tf.lite.Interpreter(model_path='./lite/model1.tflite')
i3 = tf.lite.Interpreter(model_path='./lite/model1.tflite')

s1 = i1.get_signature_runner()
s2 = i2.get_signature_runner()
s3 = i3.get_signature_runner()

y_train = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(test)])
x_train = []
test =  np.concatenate(test)
print(len(test))
for i,sign in enumerate(test):

    p1 = s1(flatten_input=np.array([sign], dtype=np.float32))
    p1 = p1[list(p1.keys())[0]]
    
    p2 = s2(flatten_input=np.array([sign], dtype=np.float32))
    p2 = p2[list(p2.keys())[0]]
    
    p3 = s3(flatten_input=np.array([sign], dtype=np.float32))
    p3 = p3[list(p3.keys())[0]]

    x_train.append(np.array([
         np.exp(p1[0])/sum(np.exp(p1[0])),
         np.exp(p2[0])/sum(np.exp(p2[0])),
         np.exp(p3[0])/sum(np.exp(p3[0]))]))    
    if i % 100 == 0:
        print(i)

x_train = np.array(x_train)
print(x_train.shape)

np.save(f"data/supermodel/x_train.npy",x_train)

x_train = np.load(f"data/supermodel/x_train.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(3, 12)),
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
tf.saved_model.save(model, 'to_lite_data/supermodel')
