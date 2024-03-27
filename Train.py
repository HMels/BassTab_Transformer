# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:42:25 2024

@author: Mels
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import mlflow

import torch
import torch.nn as nn

from Dictionary import Dictionary, save_dict
from AttentionModel import AttentionModel, save_model

##TODO write headers

#%% fuctions
def get_batch(split: str, block_size: int, batch_size: int, train_data, val_data, device: str = 'cpu') -> (torch.Tensor, torch.Tensor):
    '''
    Generate input-contexts (x) and targets (y) batch.

    Parameters
    ----------
    split : 'train' or 'val'
    block_size : The context size
    batch_size
    train_data & val_data
    device : {'cpu', 'gpu'}, optional
        Device to use. Default is 'cpu'.
    
    Returns
    -------
    x : torch.Tensor [batch x block]. Context for attention.
    y : torch.Tensor [batch x block]. Target for attention.
    '''
    
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters: int, block_size: int, batch_size: int, train_data, val_data) -> list:
    '''
    Estimates loss for training and validation data.

    Parameters
    ----------
    model : torch model
    eval_iters : Number of batches to calculate loss over.
    block_size : Number of blocks in the model.
    batch_size :
    train_data & val_data

    Returns
    -------
    out : List containing training and validation loss.
    '''
    
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size, train_data, val_data, device='cpu')
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(train_data, val_data, vocab_size: int, batch_size: int = 16, block_size: int = 32, max_iters: int = 1000, 
                eval_interval: int = 100, learning_rate: float = 1e-3, eval_iters: int = 200, n_embd: int = 128, 
                n_heads: int = 4, n_layer: int = 4, dropout: float = 0.5, show_fig: bool = True,
                patience: int = 5) -> tuple[nn.Module, dict[str, list]]:
    '''
    Trains the model on the given data with early stopping.

    Parameters
    ----------
    train_data & val_data
    vocab_size & batch_size & block_size
    max_iters : Maximum number of training iterations. Default is 1000.
    eval_interval : Interval at which to evaluate the model. Default is 100.
    learning_rate : Learning rate for the optimizer. Default is 1e-3.
    eval_iters : Number of iterations for estimating losses. Default is 200.
    n_embd & n_heads & n_layer : int, optional
    dropout : Dropout probability. Default is 0.5.
    show_fig : Whether to display the loss figure. Default is True.
    patience : Number of consecutive epochs to wait for improvement before early stopping. Default is 5.

    Returns
    -------
    model : nn.Module, The trained model.
    losses : dict, Dictionary containing estimated losses of training and evaluation datasets.
    '''
    
    # Initialize variables for early stopping
    best_val_loss = np.inf
    counter = 0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337) 
    
    m = AttentionModel(vocab_size=vocab_size, n_layer=n_layer, n_heads=n_heads,
                       n_embd=n_embd, block_size=block_size, dropout=dropout)
    model = m.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    loss_list = np.empty((max_iters//eval_interval, 2))
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, eval_iters, block_size, batch_size, train_data, val_data)
            loss_list[iter//eval_interval, 0] = losses['train']
            loss_list[iter//eval_interval, 1] = losses['val']
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Check if validation loss has improved
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                counter = 0  # Reset counter if there's improvement
            else:
                counter += 1
                
            # Early stopping condition
            if counter >= patience:
                ## TODO this can be better right?
                print(f"Validation loss has not improved for {patience} consecutive iterations. Early stopping...")
                break
    
        xb, yb = get_batch('train', block_size, batch_size, train_data, val_data, device=device)
    
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if show_fig:
        fig, ax = plt.subplots()
        ax.plot(np.linspace(1, max_iters, max_iters//eval_interval).astype(int), loss_list[:, 0], label="Training Loss")
        ax.plot(np.linspace(1, max_iters, max_iters//eval_interval).astype(int), loss_list[:, 1], label="Validation Loss")
        fig.legend()
        ax.set_yscale('log')
        fig.savefig("Results/loss_value")
    
    return model, losses


#%% load the data in the dictionary
if __name__ == "__main__":
        
    fixed_params = {
        'eval_interval': 500,
        'max_iters': 500,
        'eval_iters': 200,
        'patience': 5
    }
        
    # best params
    with open('Results/best_hyperparameters.pickle', 'rb') as f:
        params = pickle.load(f)
        params.update(fixed_params)
        
    # Convert integer choices to integers
    params['batch_size'] = int(params['batch_size'])
    params['block_size'] = int(params['block_size'])
    params['n_embd'] = int(params['n_embd'])
    params['n_heads'] = int(params['n_heads'])
    params['n_layer'] = int(params['n_layer'])
        
    ## TODO implement early stopping
    
    # Load the list back from the Pickle file
    with open('Dataset/Dataset.pickle', 'rb') as f:
        Dataset = pickle.load(f)
    
    dictionary = Dictionary(sorted(list(set(Dataset))))
        
    data = torch.tensor(dictionary.encode(Dataset), dtype=torch.long)
    
    print('vocab_size equals',dictionary.vocab_size)
    print("The data is encoded in a", data.shape[0], " size array of type",data.dtype)
    
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    # save dictionary 
    save_dict(dictionary)
    
    # initialise MLFlow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Default")
    
    # train model
    model,losses = train_model(train_data, val_data, dictionary.vocab_size, **params)
    val_loss = losses['val'].item() if isinstance(losses['val'], torch.Tensor) else losses['val']
    save_model(model)
    
    np.save("temp/train_data.npy", train_data.numpy())
    np.save("temp/val_data.npy", val_data.numpy())
    with mlflow.start_run(): # Log hyperparameters and evaluation results
        mlflow.log_params(params)
        mlflow.log_artifact("temp/train_data.npy", artifact_path="data")
        mlflow.log_artifact("temp/val_data.npy", artifact_path="data")
        mlflow.pytorch.log_model(model, 'Trained Model')
        mlflow.log_artifact('Dataset/Dataset.pickle')
        mlflow.log_metric("val_loss", val_loss)
        mlflow.set_tag("model_name","Trained Model")
    
    
    #%% test it 
    from Dataset import print_basstab
    
    #context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context = ["GDAE","||||","---2","--5-","----","----","----"]
    context = torch.reshape(torch.LongTensor(dictionary.encode(context)), shape=(len(context),1))
    print_basstab(dictionary.decode(model.generate(idx = context, max_new_tokens=100)[0].tolist()))

