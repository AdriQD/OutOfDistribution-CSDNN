# CSDNN for out-of-distribution state reconstruction


CSDNN is a protocol based on the combination of Compressed Sensing (CS) and a deep neural network model, to improve the performance of CS. The goal is twofolds:

1. Thanks to CS algorithm, we can use at most $r d\log d$ correlators (tensor produt of local Pauli operators) to estimate a quantum state with high precision. We improve the the reconstruction quality adding a deep neural network that, in a supervised learning stratgy, when the reconstruction is suboptimal.  In our job we focused on 4 qubit random Haar states, and we employ just 30-45 observables in total, instead of the theoretical upper amount of 64.

2. We want to extend the supervised denoising approach to a more general usecase, that would allow us to reconstruct states of unknwon mixedness. To achieve this goal, we make use of the out-of-distribution (OOD) paradigm. In OOd we study the generalization ablity of a network applied on a dataset of data different from the one used during training. In our case, our DNN model is trained on pure states only,  and we analyze how efficient is in reconstructring state afflicted by depolarization noise of different strength.


The git contains all the main codes used for the article realization. The **model** file is in a python notebook format, to help usage and understanding; model and other utilities are provided in the **utils** folder. Last, in the folder **article model**, the model used throghout all the experiment is provided. 





