# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:04:04 2024

Contains the dictionary class as well as the functions to save and load it. 
The dictionary class is used to encode and decode information

@author: Mels
"""

import pickle


class Dictionary():
    """
    Dictionary of the dataset. Input is the vocabulary in the form of a list. It outputs 
    an encoder and decoder function. 
    """
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        
        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(self.vocab) }
        self.itos = { i:ch for i,ch in enumerate(self.vocab) }
        
    def encode(self, string):
        # encoder: take a string, output a list of integers
        return [self.stoi[c] for c in string]
    
    def decode(self, lst):
        # decoder: take a list of integers, output a string
        return [self.itos[i] for i in lst] 
        
        
def save_dict(dictionary):
    with open(f'Dataset/Dictionary.pickle', 'wb') as file:
        pickle.dump(dictionary, file, -1) 

def load_dict():
    with open(f'Dataset/Dictionary.pickle', 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary   