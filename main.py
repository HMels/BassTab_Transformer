# -*- coding: utf-8 -*-
"""
Created on Thu Mar  12 13:00:00 2024

@author: Mels
"""
import torch

from Dictionary import  load_dict
from AttentionModel import load_model

#%% load the data
if __name__ ==  "__main__":
        ## Load the list back from the Pickle file
    #with open('Dataset.pickle', 'rb') as f:
    #    Dataset = pickle.load(f)
        
    model = load_model()
    model.eval()
    dictionary = load_dict()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # not available on Intel and AMD

    
    #%% test it 
    from Dataset import print_basstab
    
    #context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context = ["GDAE","||||","----"]
    context = torch.reshape(torch.LongTensor(dictionary.encode(context), device=device), shape=(len(context),1))
    
    for _ in range(5):
        print_basstab(dictionary.decode(model.generate(idx = context, max_new_tokens=100)[0].tolist()))


##TODO add some randomness? 
##TODO add a check to see if the answer makes sense