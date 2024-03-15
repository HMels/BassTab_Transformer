# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:11:04 2024

@author: Mels
"""

from hyperopt import fmin, tpe, hp
from functools import partial
import pickle
import torch

# Define the objective function to minimize
def objective(params):
    # Convert integer choices to integers
    params['batch_size'] = int(params['batch_size'])
    params['block_size'] = int(params['block_size'])
    params['n_embd'] = int(params['n_embd'])
    params['n_heads'] = int(params['n_heads'])
    params['n_layer'] = int(params['n_layer'])
    
    print(params)
    
    
    # Train the model with the current set of hyperparameters
    _, losses = train_model(train_data, val_data, dictionary.vocab_size, **params, show_fig=False)
    val_loss = losses['val'].item() if isinstance(losses['val'], torch.Tensor) else losses['val']
    
    # Store hyperparameters and corresponding evaluation results
    all_hyperparameters.append(params)
    all_evaluation_results.append(losses)
    return val_loss

#%%
# Constants for hyperparameters you want to keep constant
fixed_params = {
    'eval_interval': 1000,
    'max_iters': 1000,
    'eval_iters': 200
}

# Define the search space (excluding constants)
space = {
    'batch_size': hp.uniform('batch_size', 8 ,64),
    'block_size': hp.uniform('block_size', 16, 64),
    'learning_rate': hp.loguniform('learning_rate', -5, -1),
    'n_embd': hp.uniform('n_embd', 64, 256),
    'n_heads': hp.uniform('n_heads', 2, 16),
    'n_layer': hp.uniform('n_layer', 2, 8),
    'dropout': hp.uniform('dropout', 0.0, 0.5)
}

# Merge the search space with fixed parameters
space.update(fixed_params)

if __name__=="__main__":
    from Train import train_model
    from Dictionary import  load_dict
    
    # Lists to store hyperparameters and corresponding evaluation results
    global all_hyperparameters
    global all_evaluation_results
    all_hyperparameters = []
    all_evaluation_results = []
    
    dictionary = load_dict()
    
    # Load the list back from the Pickle file
    with open('Dataset.pickle', 'rb') as f:
        Dataset = pickle.load(f)
    
    data = torch.tensor(dictionary.encode(Dataset), dtype=torch.long)
    
    print('vocab_size equals',dictionary.vocab_size)
    print("The data is encoded in", data.shape, ",",data.dtype)
    
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    
    # Create a partial function for the objective with fixed parameters
    partial_objective = partial(objective)
    
    # Run hyperparameter search
    best_params = fmin(fn=partial_objective, space=space, algo=tpe.suggest, max_evals=20)
    
    print("Best hyperparameters:", best_params)

    #%%
    # Save the best hyperparameters to a pickle file
    with open('best_hyperparameters.pickle', 'wb') as f:
        pickle.dump(best_params, f)
    with open('all_hyperparameters.pickle', 'wb') as file:
        pickle.dump(all_hyperparameters, file)
    with open('all_evaluation_results.pickle', 'wb') as file:
        pickle.dump(all_evaluation_results, file)