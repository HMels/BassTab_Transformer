# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:42:25 2024

@author: Mels
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

import torch
     
from Dictionary import Dictionary, save_dict
from AttentionModel import AttentionModel, save_model

##TODO write headers

#%% fuctions
def get_batch(split, block_size, batch_size, train_data, val_data, device='cpu'):
    '''
    Generate a small batch of data of inputs x and targets y 
    
    Parameters
    ----------
    split : string
        Either train or val.
    block_size : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    
    Returns
    -------
    x : [batch x block] Tensor
        The context of the attention.
    y : [batch x block] Tensor
        The target for the attention. Block is the maximum attention length

    '''
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters, block_size, batch_size, train_data, val_data,):
    '''
    Estimates the loss for both the training and validation data. This is not the actual loss used in
    the model, but just an estimation.

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    eval_iters : int
        How many batches we want to calculate the loss over.

    Returns
    -------
    out : list
        List containing floats of the training loss and the validation loss.
        
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


def train_model(train_data, val_data, vocab_size, batch_size = 16, block_size = 32, max_iters = 1000, eval_interval = 100, learning_rate = 1e-3,
          eval_iters = 200, n_embd = 128, n_heads = 4,n_layer = 4, dropout = 0.5, show_fig=True):
    '''
    Trains the model on the given data.

    Parameters
    ----------
    train_data : torch.Tensor
        The training dataset.
    val_data : torch.Tensor
        The validation dataset.
    vocab_size : int
        The size of the vocabulary.
    batch_size : int, optional
        The number of samples in each batch. Default is 16.
    block_size : int, optional
        The size of each block. Default is 32.
    max_iters : int, optional
        The maximum number of iterations for training. Default is 1000.
    eval_interval : int, optional
        The interval at which to evaluate the model. Default is 100.
    learning_rate : float, optional
        The learning rate for the optimizer. Default is 1e-3.
    eval_iters : int, optional
        The number of iterations for estimating losses. Default is 200.
    n_embd : int, optional
        The embedding size. Default is 128.
    n_heads : int, optional
        The number of attention heads. Default is 4.
    n_layer : int, optional
        The number of layers in the model. Default is 4.
    dropout : float, optional
        The dropout probability. Default is 0.5.
    show_fig : bool, optional
        Whether to display the loss figure. Default is True.

    Returns
    -------
    model : nn.Module
        The trained model.
    losses : dict
        A dictionary containing both the estimated losses of the training and evaluation datasets.
    '''
    ##TODO MLFlow?
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # not available on Intel and AMD
    torch.manual_seed(1337) 
    print("\n------------\nTraining for: \n{batch_size:",batch_size,"\nblock_size:",block_size,"\ndropout:",
          dropout,"\neval_interval:",eval_interval,"\nlearning_rate:",learning_rate,
          "\nmax_iters:",max_iters,"\nn_embd:",n_embd,"\nn_heads:",n_heads,"\nn_layer:",n_layer,"}\n------------\n")
    
    
    m = AttentionModel(vocab_size=vocab_size, n_layer=n_layer, n_heads=n_heads,
                           n_embd=n_embd, block_size=block_size, dropout=dropout)
    model = m.to(device)
    # print the number of parameters in the model
    print("Model contains",sum(p.numel() for p in model.parameters())/1e6,'milion parameters\n')
    
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    loss_list = np.empty((max_iters//eval_interval,2))
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, eval_iters, block_size, batch_size, train_data, val_data)
            loss_list[iter//eval_interval,0]=losses['train']
            loss_list[iter//eval_interval,1]=losses['val']
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        # sample a batch of data
        xb, yb = get_batch('train', block_size, batch_size, train_data, val_data, device='cpu')
    
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if show_fig:
        fig, ax = plt.subplots()
        ax.plot(np.linspace(1, max_iters, max_iters//eval_interval).astype(int), loss_list[:,0], label="Training Loss")
        ax.plot(np.linspace(1, max_iters, max_iters//eval_interval).astype(int), loss_list[:,1], label="Validation Loss")
        fig.legend()
        ax.set_yscale('log')
        fig.savefig("loss_value")
    
    return model, losses


#%% load the data in the dictionary
if __name__ == "__main__":
        
    # Load the list back from the Pickle file
    with open('Dataset.pickle', 'rb') as f:
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
    
    # best params
    with open('best_hyperparameters.pickle', 'rb') as f:
        best_params = pickle.load(f)
    
    # train model
    model,_ = train_model(train_data, val_data, dictionary.vocab_size, **best_params)
    save_model(model)
    
    
    
    #%% test it 
    from Dataset import print_basstab
    
    #context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context = ["GDAE","||||","---2","--5-","----","----","----"]
    context = torch.reshape(torch.LongTensor(dictionary.encode(context)), shape=(len(context),1))
    print_basstab(dictionary.decode(model.generate(idx = context, max_new_tokens=100)[0].tolist()))

