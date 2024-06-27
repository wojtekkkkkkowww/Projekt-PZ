import numpy as np
import pandas as pd
from tabulate import tabulate

models = ['model1', 'model2', 'model3', 'model1_lite', 'model2_lite', 'model3_lite', 'supermodel_lite', 
          'model1_lite_permuted', 'model2_lite_permuted', 'model3_lite_permuted', 'supermodel_lite_permuted']
short_names = ['m1', 'm2', 'm3', 'm1L', 'm2L', 'm3L', 'supermodelL', 'm1LP', 'm2LP', 'm3LP', 'supermodelLP']

name_dict = dict(zip(models, short_names))

data = pd.DataFrame(columns=['Model', 'Accuracy', 'Sureness', 'Time'])

for model in models:
    accuracies = np.load(f'results/{model}/accuracies.npy')
    sureness_values = np.load(f'results/{model}/sureness_values.npy')
    elapsed_time = np.load(f'results/{model}/time.npy')

    last_accuracy = accuracies[-1]
    last_sureness = sureness_values[-1]
    last_time = elapsed_time[-1]
    data = data._append({'Model': name_dict[model], 'Accuracy': last_accuracy, 'Sureness': last_sureness, 'Time': last_time}, ignore_index=True)

latex_table = tabulate(data, tablefmt='latex', headers='keys', showindex=False)

print(latex_table)