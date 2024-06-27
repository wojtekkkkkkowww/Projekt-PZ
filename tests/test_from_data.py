import numpy as np
import tensorflow as tf
import argparse
import os
import time

parser = argparse.ArgumentParser(prog='Test models')
parser.add_argument('-p', '--permuted', action='store_true')
parser.add_argument('-l', '--lite', action='store_true')
parser.add_argument('-s', '--sequential', action='store_true')
parser.add_argument('-m', '--model')
parser.add_argument('-d', '--data') # data/ASLtrain

def gen_seq(sequence_length, X_train):
    sequences = []
    for i in range(0, len(X_train) - sequence_length + 1):
        sequence = np.array(X_train[i:i + sequence_length]).reshape(sequence_length,21*3)
        sequences.append(sequence)
    return sequences

def get_dataset(data_dir,sequential,permuted):
    train = []
    for i in range(NUMBER_OF_SYMBOLS):
        loaded = np.load(os.path.join(data_dir, f'{i}.npy'))

        if(permuted):
            key = np.load(f'keys/{MODEL_NAME.split(".")[0]}_key.npy')
            x_perm = np.array([np.array(x.flatten()[key]).reshape(21,3) for x in loaded])
            loaded = x_perm

        if(sequential):
            loaded = gen_seq(10,loaded)

        train.append(loaded)

    x_train =  np.concatenate(train)
    y_train = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(train)])
    
    return x_train,y_train


if __name__ == "__main__":
    args = parser.parse_args()
    if None in [args.data, args.model]:
        print("see -h")
        exit()
    
    MODEL_NAME = args.model
    NUMBER_OF_SYMBOLS = len(os.listdir(f'{os.getcwd()}/{args.data}'))
    x_test, y_test = get_dataset(args.data,args.sequential,args.permuted)

    MODEL_LITE = False
    if(args.model.endswith('tflite')):
        interpreter = tf.lite.Interpreter(model_path=f'models/{args.model}')
        signature = interpreter.get_signature_runner()
        MODEL_LITE = True
    else:
        model = tf.keras.models.load_model(f"models/{args.model}")

    SUPER = False
    if(args.model.startswith('supermodel')):
        if(args.lite):
            interpreters = [tf.lite.Interpreter(model_path=f'models/model{i}.tflite') for i in range(1,4)]
            signatures = [i.get_signature_runner() for i in interpreters]
        else:
            models = [tf.keras.saving.load_model(f"models/model{i}.keras") for i in range(1,4)]
        SUPER = True
        print("SUPER")

    sureness_values = []
    correct_predictions = 0
    accuracies = []
    elapsed_time = []

    for i, sample in enumerate(x_test):
        startTime = time.time()

        if(SUPER):
            res = []
            for m in range(3):
                if(args.lite):
                    atr = list(signatures[m].get_input_details().keys())[0]
                    print(sample.shape)
                    p = signatures[m](**{atr:np.array([sample], dtype=np.float32)})
                    p = p[list(p.keys())[0]]
                else:
                    p = models[m].predict(sample)

                res.append(p)
            sample = np.array(np.concatenate(res))

        if(MODEL_LITE):
            atr = list(signature.get_input_details().keys())[0]
            predictions = signature(**{atr:np.array([sample], dtype=np.float32)})
            predictions = predictions[list(predictions.keys())[0]]
        else:
            predictions = model.predict(np.array([sample]),verbose='false')

        predicted_sign_index = np.argmax(predictions[0])
        sureness = predictions[0][predicted_sign_index]
        
        elapsed_time.append(time.time() - startTime)

        sureness_values.append(sureness)
        if predicted_sign_index == y_test[i]:
            correct_predictions += 1
        accuracy = correct_predictions / (i + 1)
        print(f"{i}/{len(x_test)} || acc : {accuracy}")
        accuracies.append(accuracy)



    # Create the directory if it doesn't exist
    
    model_name = args.model.split(".")[0] if not MODEL_LITE else args.model.split(".")[0] + "_lite"
    model_name = model_name + "_permuted" if args.permuted else model_name

    directory = f'results/{model_name}'
    os.makedirs(directory, exist_ok=True)

    # Save accuracies and sureness values to files
    np.save(f'{directory}/accuracies.npy', accuracies)
    np.save(f'{directory}/sureness_values.npy', sureness_values)
    np.save(f'{directory}/time.npy', elapsed_time)