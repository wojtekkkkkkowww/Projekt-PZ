import numpy as np
import tensorflow as tf
import argparse
import os
import time

parser = argparse.ArgumentParser(prog='Test models')
parser.add_argument('-p', '--permuted', action='store_true')
parser.add_argument('-l', '--lite', action='store_true')
parser.add_argument('-m', '--model')
parser.add_argument('-d', '--data') # data/ASLtrain

def get_test_dataset(data_dir):
    test = [np.load(os.path.join(data_dir, f'{i}.npy')) for i in range(NUMBER_OF_SYMBOLS)]
    x_test =  np.concatenate(test) 
    y_test = np.concatenate([np.full(len(sign), i) for i, sign in enumerate(test)])
    print(x_test.shape)
    print(y_test)
    return x_test, y_test


if __name__ == "__main__":
    args = parser.parse_args()
    if None in [args.data, args.model]:
        print("see -h")
        exit()
    
    NUMBER_OF_SYMBOLS = len(os.listdir(f'{os.getcwd()}/{args.data}'))
    x_test, y_test = get_test_dataset(args.data)

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
                if(args.permuted):
                    key = np.load(f'keys/model{m+1}_key.npy')
                    sample = np.array(sample.flatten()[key]).reshape(21,3)

                if(args.lite):
                    atr = list(signatures[m].get_input_details().keys())[0]
                    print(sample.shape)
                    p = signatures[m](**{atr:np.array([sample], dtype=np.float32)})
                    p = p[list(p.keys())[0]]
                else:
                    p = models[m].predict(sample)

                res.append(p)
            sample = np.array(np.concatenate(res))
        elif(args.permuted):
            key = np.load(f'keys/{args.model.split(".")[0]}_key.npy')
            sample = np.array(sample.flatten()[key]).reshape(21,3)

        if(MODEL_LITE):
            atr = list(signature.get_input_details().keys())[0]
            predictions = signature(**{atr:np.array([sample], dtype=np.float32)})
            predictions = predictions[list(predictions.keys())[0]]
        else:
            predictions = model.predict(np.array([sample]))

        predicted_sign_index = np.argmax(predictions[0])
        sureness = predictions[0][predicted_sign_index]
        
        elapsed_time.append(time.time() - startTime)

        sureness_values.append(sureness)
        if predicted_sign_index == y_test[i]:
            correct_predictions += 1
        accuracy = correct_predictions / (i + 1)
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