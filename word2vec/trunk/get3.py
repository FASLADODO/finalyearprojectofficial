import os
files=[i for i in os.listdir("/home/rs6713/Documents/finalyearprojectofficial/word2vec/trunk/matlab_sentence_vectors") if i.find("norm3")!=-1 and i.find("size100")!=-1]
print files
