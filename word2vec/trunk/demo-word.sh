make
#if [ ! -e word-descriptions.txt ]; then
sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < descriptions.txt | tr "A-Z,.'_" "a-z " | tr "A-Z" "a-z" > word-descriptions.txt
#fi

#sed -e is an editing intruction, used when mult instrs
#substitute a/b b for first inst a in line
#-c replace SET1 with its compliment/set2 (all not in SET1) tr SET1 SET2
# 0-9'_ \n
#time ./word2vec -train word-descriptions.txt -output vectors.bin -cbow 1 -size 200 -window 8 -negative 20 -hs 0 -sample 0 -threads 20 -binary 1 -iter 15
time ./word2vec -train word-descriptions.txt -output vectors.txt -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15 
time ./word2vec -train word-descriptions.txt -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15 -save-vocab word-vocab.txt
./distance vectors.bin

#size is layer1_size 200, 785 UNIQUE WORDS 157000
#cbow is if(cbow) train cbow architecture
#window is how many words to search to r/l of
#negative says how many weights to tune per word 5-20 recommended for small datasets
#hs is whether using hierarchal softmax for training
#sample is subsampling, set 0 to disable, smaller the less sample frequent words
#binary if binary save in binray moded
#iter, number training iterations
#threads is parallelism

#write for all vocab for all layersize write to fo, so need xbyy grid

#Parameter sample controls how much subsampling occurs, smaller the less sample
#Set 'sample' to 0 to disable subsampling.
#Negative sampling addresses this by each training samply only modify small % weights. 5-20 recommended for small data sets
#window is how many words nearby
#space tab, EOL assumed to be word boundaries
#marked end of sentence </s> is word empty and char new line
#words with fewer than wordcount are removed from training text
#sort vocab by number of occurences in descending order

# * Create binary Huffman tree using the word counts.
# * Frequent words will have short unique binary codes.
# * Huffman encoding is used for lossless compression.
# * The vocab_word structure contains a field for the 'code' for the word.
#binary codes assigned to each word
#build vocabulary from training file, hash table built allows fast lookup of word vector to vocab_word obj

# // The special token </s> is used to mark the end of a sentence. In training,
#// the context window does not go beyond the ends of a sentence.
#if vocab too large, trim modst infrequent words
#he vocabulary is considered "too large" when it's filled more
#    // than 70% of the hash table (this is to try and keep hash collisions
#// down).
#Report the final vocabulary size, and the total number of words 
#// (excluding those filtered from the vocabulary) in the training set
