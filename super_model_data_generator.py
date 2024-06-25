import numpy as np
import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser(prog='Data generator for super model')
parser.add_argument('-l', '--lite', action='store_true')
parser.add_argument('-d', '--data')

if __name__ == "__main__":
    args = parser.parse_args()
    ext = "keras" if(args.lite) else "tflite"
    interpreters = [tf.lite.Interpreter(model_path=f'./model{i}.{ext}') for i in range(3)]
    signatures = [i.get_signature_runner() for i in interpreters]

    path = f'data/model1/{args.data}'
    signs_num = len(os.listdir(f'{os.getcwd()}/{path}'))

    for i in range(signs_num):
        data = []
        for m in range(3):
            data.append(np.load(os.path.join(f'data/model{m}/{args.data}', f'{i}.npy')))
        data = np.concatenate(data)

        x_train = []
        for sign in data:
            res = []
            for s in signatures:
                p = s(flatten_input=np.array([sign], dtype=np.float32))
                p = p[list(p.keys())[0]]
                res.append(np.exp(p[0])/sum(np.exp(p[0])))
            x_train.append(np.array(res))
        x_train = np.concatenate(x_train)
        np.save(f"data/supermodel/{i}.npy",x_train)    


    
