import numpy as np
import tensorflow as tf

NUMBER_OF_SYMBOLS = 12
EPOCH = 50

files = [f'data/model2/{i}.npy' for i in range(0, NUMBER_OF_SYMBOLS)]
data = [np.load(file) for file in files]

x_train =  np.concatenate(data)
y_train = np.array([label for label in range(0,NUMBER_OF_SYMBOLS) for cords in data[label]])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 3)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(NUMBER_OF_SYMBOLS, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=EPOCH)

model.save('models/model2.keras')
