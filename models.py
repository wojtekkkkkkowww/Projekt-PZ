import numpy as np
import tensorflow as tf
import os
import argparse
import keras



parser = argparse.ArgumentParser(prog='Model trainer')
parser.add_argument('-d', '--data')
parser.add_argument('-s', '--sequential', action='store_true')
parser.add_argument('-m', '--model')
parser.add_argument('-e', '--epoch')
parser.add_argument('-p', '--permuted',action='store_true') 

args = parser.parse_args()
NUMBER_OF_SYMBOLS = len(os.listdir(f'{os.getcwd()}/{args.data}'))
MODEL_NAME = args.model
SEQ_LEN = 10


model1 = keras.Sequential([
    keras.layers.InputLayer(input_shape=(SEQ_LEN, 21*3)),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(64),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(NUMBER_OF_SYMBOLS, activation='softmax')
])

model2 = keras.Sequential([
    keras.layers.InputLayer(input_shape=(SEQ_LEN, 21*3)),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(64),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(NUMBER_OF_SYMBOLS, activation='softmax')
])

model3 = keras.Sequential([
    keras.layers.Flatten(input_shape=(21,3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(NUMBER_OF_SYMBOLS, activation='softmax')
])

supermodel = keras.Sequential([
    keras.layers.Flatten(input_shape=(3, NUMBER_OF_SYMBOLS)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(NUMBER_OF_SYMBOLS, activation='softmax')
])

def gen_seq(X_train):
    sequences = []
    for i in range(0, len(X_train) - SEQ_LEN + 1):
        sequence = np.array(X_train[i:i + SEQ_LEN]).reshape(SEQ_LEN,21*3)
        sequences.append(sequence)
    return sequences

def get_dataset(data_dir,sequential,permuted):
    train = []
    for i in range(NUMBER_OF_SYMBOLS):
        loaded = np.load(os.path.join(data_dir, f'{i}.npy'))

        if(permuted):
            key = np.arange(21*3)
            np.random.shuffle(key)
            np.save(f'keys/{MODEL_NAME}_key.npy', key)
            x_perm = np.array([np.array(x.flatten()[key]).reshape(21,3) for x in loaded])
            loaded = x_perm

        if(sequential):
            loaded = gen_seq(loaded)

        train.append(loaded)

    x_train =  np.concatenate(train)
    y_train = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(train)])
    
    return x_train,y_train

def get_model(model):
    return {
        'model1':model1,
        'model2':model2,
        'model3':model3,
        'supermodel':supermodel
    }[model]

def save_model(epoch, x_train, y_train):
    model = get_model(MODEL_NAME)

    model.compile(optimizer='adam',
             loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=int(epoch))

    model.save(f'models/{MODEL_NAME}.keras')
    model.export(f'to_lite_data/{MODEL_NAME}')

if __name__ == "__main__":
    if None in [args.data, args.model, args.epoch]:
        print("see -h")
        exit()
    
    x_train, y_train = get_dataset(args.data,args.sequential,args.permuted)
    save_model(args.epoch, x_train, y_train)