import numpy as np
import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser(prog='Data generator for super model')
parser.add_argument('-l', '--lite', action='store_true')
parser.add_argument('-p', '--permuted', action='store_true')
parser.add_argument('-d', '--data')

if __name__ == "__main__":
    args = parser.parse_args()
    if(args.data is None):
        print("see -h")
        exit()

    if(args.lite):
        interpreters = [tf.lite.Interpreter(model_path=f'models/model{i}.tflite') for i in range(1,4)]
        signatures = [i.get_signature_runner() for i in interpreters]
    else:
        models = [tf.keras.saving.load_model(f"models/model{i}.keras") for i in range(1,4)]
    

    path = f'data/model1/{args.data}'
    signs_num = len(os.listdir(f'{os.getcwd()}/{path}'))

    for i in range(signs_num):
        print(f"ZNAK : {i}")
        data = np.concatenate([np.load(os.path.join(f'data/model{m}/{args.data}', f'{i}.npy')) for m in range(1,4)])

        x_train = []
        for i,sign in enumerate(data):
            print(i,len(data))
            res = []
            for m in range(3):
                if(args.permuted):
                    key = np.load(f'keys/model{m+1}_key.npy')
                    sign = np.array(sign.flatten()[key]).reshape(21,3)

                if(args.lite):
                    atr = list(signatures[m].get_input_details().keys())[0]
                    p = signatures[m](**{atr:np.array([sign], dtype=np.float32)})
                    p = p[list(p.keys())[0]]
                else:
                    p = models[m].predict(np.array([sign]),verbose=0)
                    
                res.append(p)
            x_train.append(res)
        np.save(f"data/supermodel/{args.data}/{i}.npy",np.array(x_train))    


    
