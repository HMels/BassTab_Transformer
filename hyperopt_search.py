# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:11:04 2024

@author: Mels
"""

from hyperopt import fmin, tpe, hp
from functools import partial
import pickle
import torch
import mlflow
import numpy as np

# Define the objective function to minimize
def objective(params : dict):
    # Convert integer choices to integers
    params['batch_size'] = int(params['batch_size'])
    params['block_size'] = int(params['block_size'])
    params['n_embd'] = int(params['n_embd'])
    params['n_heads'] = int(params['n_heads'])
    params['n_layer'] = int(params['n_layer'])   
    
    with mlflow.start_run(): # Log hyperparameters and evaluation results
        # Train the model with the current set of hyperparameters
        model, losses = train_model(train_data, val_data, dictionary.vocab_size, **params, show_fig=False)
        val_loss = losses['val'].item() if isinstance(losses['val'], torch.Tensor) else losses['val']

    
        np.save("temp/train_data_hyperopt.npy", train_data.numpy())
        np.save("temp/val_data_hyperopt.npy", val_data.numpy())
        for key, value in params.items():
            mlflow.log_param(key, int(value))
        mlflow.log_params(fixed_params)
        mlflow.log_artifact("temp/train_data_hyperopt.npy", artifact_path="data")
        mlflow.log_artifact("temp/val_data_hyperopt.npy", artifact_path="data")
        mlflow.pytorch.log_model(model, 'Test Model in HyperOpt Run')
        mlflow.log_metric("val_loss", val_loss)
        mlflow.set_tag("model_name","HyperOpt Run")
    
    return val_loss

#%%
# Constants for hyperparameters you want to keep constant
fixed_params = {
    'eval_interval': 500,
    'max_iters': 500,
    'eval_iters': 200,
    'patience': 5
}

# Define the search space (excluding constants)
space = {
    'batch_size': hp.quniform('batch_size', 10 ,20, q=1),
    'block_size': hp.quniform('block_size', 16, 64, q=1),
    'learning_rate': hp.loguniform('learning_rate', -5, -1),
    'n_embd': hp.quniform('n_embd', 64, 256, q=1),
    'n_heads': hp.quniform('n_heads', 2, 16, q=1),
    'n_layer': hp.quniform('n_layer', 2, 8, q=1),
    'dropout': hp.uniform('dropout', 0.0, 0.5)
}

# Merge the search space with fixed parameters
space.update(fixed_params)

if __name__=="__main__":
    from Train import train_model
    from Dictionary import  load_dict
    
    dictionary = load_dict()
    
    # Load the list back from the Pickle file
    with open('Dataset/Dataset.pickle', 'rb') as f:
        Dataset = pickle.load(f)
    
    data = torch.tensor(dictionary.encode(Dataset), dtype=torch.long)
    
    print('vocab_size equals',dictionary.vocab_size)
    print("The data is encoded in", data.shape, ",",data.dtype)
    
    N1 = int(0.25*len(data)) # first 90% will be train, rest val
    N2 = int(0.3*len(data)) # first 90% will be train, rest val
    train_data = data[:N1]
    val_data = data[N1:N2]    
    
    # initialise MLFlow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("HyperOpt Run")
    
    # Create a partial function for the objective with fixed parameters
    partial_objective = partial(objective)
    
    # Run hyperparameter search
    best_params = fmin(fn=partial_objective, space=space, algo=tpe.suggest, max_evals=4)
    print("Best hyperparameters:", best_params)

    #%%
    # Save the best hyperparameters to a pickle file
    with open('Results/best_hyperparameters.pickle', 'wb') as f:
        pickle.dump(best_params, f)