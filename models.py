import numpy as np
import tensorflow as tf
import os
import argparse



parser = argparse.ArgumentParser(prog='Model trainer')
parser.add_argument('-d', '--data')
parser.add_argument('-s', '--sequential', action='store_true')
parser.add_argument('-m', '--model')
parser.add_argument('-e', '--epoch')
parser.add_argument('-p', '--permuted',action='store_true') 

args = parser.parse_args()
NUMBER_OF_SYMBOLS = len(os.listdir(f'{os.getcwd()}/{args.data}'))


model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(32, activation='relu'),
])

model2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(10, 21*3)),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
])

model3 = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(21, 3)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
])

supermodel = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(3, NUMBER_OF_SYMBOLS)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
])

def gen_seq(sequence_length, X_train):
    sequences = []
    for i in range(0, len(X_train) - sequence_length + 1):
        sequence = np.array(X_train[i:i + sequence_length]).reshape(sequence_length,21*3)
        sequences.append(sequence)
    return sequences

def get_dataset(data_dir,sequential):
    train = []
    for i in range(NUMBER_OF_SYMBOLS):
        loaded = np.load(os.path.join(data_dir, f'{i}.npy'))
        if(sequential):
            loaded = gen_seq(10,loaded)
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

def save_model(model_string, epoch, x_train, y_train, permuted):
    if(permuted):
        key = np.arange(21*3)
        np.random.shuffle(key)
        np.save(f'keys/{model_string}_key.npy', key)
        x_perm = np.array([np.array(x.flatten()[key]).reshape(21,3) for x in x_train])
        x_train = x_perm

    model = get_model(model_string)
    model.add(tf.keras.layers.Dense(NUMBER_OF_SYMBOLS))
    model.add(tf.keras.layers.Softmax())

    model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=int(epoch))

    model.save(f'models/{model_string}.keras')
    tf.saved_model.save(model, f'to_lite_data/{model_string}')

if __name__ == "__main__":
    if None in [args.data, args.model, args.epoch]:
        print("see -h")
        exit()
    
    x_train, y_train = get_dataset(args.data,args.sequential)
    save_model(args.model, args.epoch, x_train, y_train, args.permuted)