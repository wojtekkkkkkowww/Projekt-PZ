import numpy as np
import tensorflow as tf

NUMBER_OF_SYMBOLS = 12
EPOCH = 200

files = [f'data/model2/{i}.npy' for i in range(0, NUMBER_OF_SYMBOLS)]
data = [np.load(file) for file in files]

x_train =  np.concatenate(data)
y_train = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(data)])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 3)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(NUMBER_OF_SYMBOLS)
])



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=EPOCH)

model.save('models/model2.keras')
tf.saved_model.save(model, 'to_lite_data/model2')
