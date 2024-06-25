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
        data = []
        for m in range(1,4):
            d = np.load(os.path.join(f'data/model{m}/{args.data}', f'{i}.npy'))
            if(args.permuted):
                key = np.load(f'keys/model{m}_key.npy')
                d_perm = np.array([np.array(x.flatten()[key]).reshape(21,3) for x in d])
                d = d_perm
            data.append(d)

        data = np.concatenate(data)

        x_train = []
        for sign in data:
            res = []
            for i in range(3):
                if(args.lite):
                    p = signatures[i](flatten_input=np.array([sign], dtype=np.float32))
                    p = p[list(p.keys())[0]]
                else:
                    p = models[i].predict(np.array([sign]),verbose=0)
                    
                res.append(p)

        x_train = np.array(res)
        np.save(f"data/supermodel/{args.data}/{i}.npy",x_train)    


    
