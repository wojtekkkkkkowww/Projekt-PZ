import numpy as np
import tensorflow as tf

NUMBER_OF_SYMBOLS = 12
EPOCH = 50

def get_y_train(data):
    out = []
    for i in range(0, NUMBER_OF_SYMBOLS):
        for j in data[i]:
            out.append(i)
    return out 
filenames = [f'data/model3/{i}.npy' for i in range(0, NUMBER_OF_SYMBOLS)]
data = [np.load(filename) for filename in filenames]


x_train =  np.concatenate(data) # lista macierzy o rozmiarze (21, 3)
y_train = np.array(get_y_train(data)) # liczba który symbol
print(np.shape(x_train), np .shape(y_train))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUMBER_OF_SYMBOLS)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=EPOCH)

model.save('model3.keras')