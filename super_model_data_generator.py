import numpy as np
import tensorflow as tf
import os
import argparse
from collections import deque
from models import GestureRecognizerModel,sequentials

parser = argparse.ArgumentParser(prog='Data generator for super model')
parser.add_argument('-l', '--lite', action='store_true')
parser.add_argument('-p', '--permuted', action='store_true')
parser.add_argument('-d', '--data')

SEQ_LEN = 10


if __name__ == "__main__":
    args = parser.parse_args()
    if(args.data is None):
        print("see -h")
        exit()

    seq_data = deque([np.zeros((21*3)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    submodels = [GestureRecognizerModel(None,f'model{i}',args.permuted,sequentials[i-1],args.lite) for i in range(1,4)]

    NUMBER_OF_SYMBOLS = len(os.listdir(f'{os.getcwd()}/data/model1/{args.data}'))

    for i in range(NUMBER_OF_SYMBOLS):
        print(f"ZNAK : {i}")
        data = np.concatenate([np.load(os.path.join(f'data/model{m}/{args.data}', f'{i}.npy')) for m in range(1,4)])
        
        x_train = []
        for sign in data:
            res = []

            for subm in submodels:
                res.append( subm.predict(sign) )

            x_train.append(np.concatenate(res))
        print(np.array(x_train).shape)
        os.makedirs(f'data/supermodel/{args.data}', exist_ok=True)
        np.save(f"data/supermodel/{args.data}/{i}.npy",np.array(x_train))    


    
