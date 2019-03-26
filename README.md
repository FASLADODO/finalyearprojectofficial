# MEng Electronics and Information Final Year Project

Learning the associations between pedestrian images and their NL descriptions.

## Tech Stack

Python scripts were used to automate dataset augmentation, the running of MATLAB networks under different parameters, and word2vec sentence embeddings generation.

NL Embeddings were generated using Word2Vec on the descriptive dataset, a number of settings were trialled with their effectiveness compared using a combination of Word2Vecs inbuilt scoring functions and visualisations of semantic words proximity in vector space via a comination of matplotlib and 

Image features generated using a variety of methods:
* LOMO
* Alexnet + Transfer Learning
* AutoEncoders

Image and NL Sentence features are concatenated channel wise and fed into a convolutional neural network that produces a binary match score.

