# BassTabAI_v2

This model is a simplified version of the BassTabAI model. In this version, we will use a deepened understanding of Neural Networks to create a NLP that focuses more on the Attention Mechanism (in this case Self-Attention).

## The Architecture

1. **Dataset**
    - **Dataset Preparation**:
        - Dataset starts with webscraping all the data. It then ensures everything is in the right format, such as pull-offs and hammer-ons being denoted by both the lowercase p and h. It then ensures all tabs start in the same way and splits the bars.
    - **Tab Recognition**:
        - The model then tries to recognize the actual tabs from text that surrounds it. It does so by focusing on the first item per line, which should be G, D, A, or E.
    - **Tokenization**:
        - The data is split into tokens, which are just the notes that are played per beat.

        For example:

        ```
        G|---9h11  
        D|-------  
        A|-7-----  
        E|-------
        ```

        Is split into `('GDAE', '||||', '----', '--7-', '----', '9---', 'h---', '1---', '1---')`

        Initially, I thought it would be better to split the data per actual note (so 9 would become 9h and 1 and 1 would be added together to form 11). But for now, this approach seems to work.
    - **Tab Printing**:
        - The file also contains the `print_basstab` function, which takes a list of tokens and then prints them as a tab (the inverse of the previous example).

2. **Training**:
    - Train is used to train the model on the dataset, which has been saved as Dataset.pickle.
    - **Dictionary Generation**:
        - Dictionary is generated, which contains both the encoder and decoder for the data.
    - **Data Preparation**:
        - Data is split into training and validation, as well as different batches.
    - **Loss Estimation**:
        - `estimate_loss` function is used to estimate the loss as an output during training.

4. **Attention Model**:
    - In `AttentionModel` contains the actual model architecture. This one is split into different blocks that are inspired by the GPT framework:
        - **Block Structure**:
            - MultiHeadAttention
            - LayerNorm + Residue
            - FeedForward (with ReLu and Dropout)
            - LayerNorm + Residue
        - **Model Structure**:
            - Token embedding
            - Position embedding (Attention does not remember the position of tokens, so these need to be programmed in)
            - A N amount of Blocks
            - LayerNorm
            - Linear Layer for output
        - **Forward Pass**:
            - The forward of this model is the cross-entropy.

5. **Model.py**:
    - `Model.py` contains the example to output the model.

6. **Hyperparameter Optimization**:
    - `hyperopt_search.py` contains the function that uses hyperopt to research the best hyperparameters, which are stored in a pickle file.

## Training the Model

When training the model with the optimized hyperparameters:

```json
{
    "batch_size": 48,
    "block_size": 52,
    "dropout": 0.09853040756166137,
    "eval_interval": 1000,
    "eval_iters": 200,
    "learning_rate": 0.007617883451530519,
    "max_iters": 1000,
    "n_embd": 85,
    "n_heads": 7,
    "n_layer": 4
}
} 

we can see the next loss curve for both training and validation:
Loss Curve

![Loss Curve](loss_value.png)

An example of what the code returns now:

runcell(5, 'C:/Users/Mels/Documents/GitHub/BassTab_Transformer/Model.py')
G|---4-4-4-4-4-4-4---4------------|-----6--------------------------------------------
D|-----------------4----2--2------|---4-------4-------------------1------0-----------
A|-2----------------------------2-|-4-----4-----------------2--------12-2--5--10-10-|
E|--------------------------------|---------------2--7--9-|-3-5---1-0---0--70-------|

Things To Do

    Implement MLFlow better
    Make the model more complex to generate better tabs
    Make the model faster by implementing C++
