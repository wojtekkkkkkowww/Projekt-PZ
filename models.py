import numpy as np
import tensorflow as tf
import os
import argparse
import keras
from collections import deque

sequentials = [True, True, False]


class GestureRecognizerModel():
    def __init__(self, model, name, permuted, sequential,lite=True, seq_len=10):
        self.model = model
        self.name = name

        self.key = None
        self.signature = None
        self.seq_data = None

        if(model is None):
            self.load_model(lite)

        if(sequential):
            self.seq_data = deque([np.zeros((21*3)) for _ in range(seq_len)], maxlen=seq_len)

        if(permuted):
            self.key = np.load(f'keys/{self.name}_key.npy')


    def load_model(self, lite):
        if(lite):
            interpreter = tf.lite.Interpreter(model_path=f'models/{self.name}.tflite')
            self.signature = interpreter.get_signature_runner()
        else:
            self.model = keras.saving.load_model(f"models/{self.name}.keras")

    def predict(self, sign):
        if(self.key):
            sign = np.array(sign.flatten()[self.key]).reshape(21,3)

        if(self.seq_data):
            self.seq_data.appendleft(sign.flatten())
            sign = np.array(self.seq_data)

        if(self.signature):
            atr = list(self.signature.get_input_details().keys())[0]
            p = self.signature(**{atr:np.array([sign], dtype=np.float32)})
            return p[list(p.keys())[0]]
        else:
            return self.model.predict([sign])


    def train_save(self, epoch, x_train, y_train):
        self.model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=int(epoch))

        self.model.save(f'models/{MODEL_NAME}.keras')
        self.model.export(f'to_lite_data/{MODEL_NAME}')


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

if __name__ == "__main__":
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

    if None in [args.data, args.model, args.epoch]:
        print("see -h")
        exit()

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
    
    get_model = lambda model : {
        'model1':model1,
        'model2':model2,
        'model3':model3,
        'supermodel':supermodel}[model]

    x_train, y_train = get_dataset(args.data,args.sequential,args.permuted)
    model = GestureRecognizerModel(get_model(args.model),args.model,args.permuted,args.sequential,args)
    model.train_save(args.epoch, x_train, y_train)
