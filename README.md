# CSDNN for out-of-distribution state reconstruction


CSDNN is a protocol based on the combination of Compressed Sensing (CS) and a deep neural network model, to improve the performance of CS. The goal is twofolds:

1. Thanks to CS algorithm, we can use at most $r d\log d$ correlators (tensor produt of single qubit Pauli operators) to estimate a quantum state with high precision. We improve the CS algorithm adding a deep neural network that trains on the CS estimation outputs, in a supervised learning strategy, and improve them. In our job, we focused on 4 qubit random Haar states and employ 30-45 correlators only, instead of the CS theoretical upper bound of 64, or the full informational amount of 256.

2. We want to extend the supervised denoising approach to a more general usecase, that would allow us to reconstruct states of unknwon mixedness witout any prior information about them. To achieve this goal, we make use of the out-of-distribution (OOD) paradigm. In OOD we study the generalization ability of a network by using it in inference on data different from the ones used during training. In our case, a quantum state estimation task, our DNN model is trained on pure states only, and we analyze how efficient it is in reconstructring state afflicted by depolarization noise of different strength.

## Article link

All information and details can be found in the article []()

## REPO structure
- The git contains all the main codes used for the article realization. In the _AttentionModel.ipynb_ file, the whole train (valid)-test of the models is provided in a python notebook, to make its use and understanding handier; the model class and other utilities can be found in the **/utils**. In **/fixed point inference** a .py file for inferring the OOD esitmations with the fixed point strategy, as described in the article.

 - Last, in the folder **/article model**, a .ph file of the model used throghout all the experiment is provided. In **/CS analysis** the code to produce Fig.5. 


## Dependencies

- torch 2.0.1
- cvxpy  1.4.
- cuda-version 12.0          
- cudnn   8.8.0.121
- mosek  10.1.20
- pandas 2.1.3
- tqdm  4.66.1

![image](/background/br.png)
(**painting, Brecht Evens**)