import numpy as np
import tensorflow as tf
import argparse
import os
import time

from models import GestureRecognizerModel,sequentials

parser = argparse.ArgumentParser(prog='Test models')
parser.add_argument('-p', '--permuted', action='store_true')
parser.add_argument('-l', '--lite', action='store_true')
parser.add_argument('-s', '--sequential', action='store_true')
parser.add_argument('-m', '--model')
parser.add_argument('-d', '--data') # data/ASLtrain

SEQ_LEN = 10

def get_dataset(data_dir):
    train = [np.load(os.path.join(data_dir, f'{i}.npy')) for i in range(NUMBER_OF_SYMBOLS)]
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
    x_test, y_test = get_dataset(args.data)

    LITE = False
    if(args.model.endswith('tflite')):
        LITE = True

    model = GestureRecognizerModel(None,args.model.split(".")[0],args.permuted,args.sequential,LITE)

    SUPER = False
    if(args.model.startswith('supermodel')):
        SUPER = True
        submodels = [GestureRecognizerModel(None,f'model{i}',args.permuted,sequentials[i-1],args.lite) for i in range(1,4)]

    sureness_values = []
    correct_predictions = 0
    accuracies = []
    elapsed_time = []

    for i, sample in enumerate(x_test):
        startTime = time.time()

        if(SUPER):
            res = []
            for subm in submodels:
                res.append( subm.predict(sample) )
            
            sample = np.array(np.concatenate(res))

        predictions = model.predict(sample)

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
    
    model_name = args.model.split(".")[0] if not LITE else args.model.split(".")[0] + "_lite"
    model_name = model_name + "_permuted" if args.permuted else model_name

    directory = f'results/{model_name}'
    os.makedirs(directory, exist_ok=True)

    # Save accuracies and sureness values to files
    np.save(f'{directory}/accuracies.npy', accuracies)
    np.save(f'{directory}/sureness_values.npy', sureness_values)
    np.save(f'{directory}/time.npy', elapsed_time)