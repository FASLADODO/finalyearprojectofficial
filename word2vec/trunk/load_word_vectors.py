import numpy as np
'''
SOME PHRASES HAVE "none" dewscription need to pre-process this out
'''
'''
import numpy as np
def load_bin_vec(fname):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)  
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
    return word_vecs
'''



#Load all word vector associations

file_name="vectors.txt"
word=[] #Contains key of words
vector=[] #Contains 
with open(file_name, "r") as f:
	header=f.readline()
	vocab_size, vector_size = map(int, header.split())
	print vocab_size 
	print vector_size

	for l in range(vocab_size):
		line=f.readline()
		w,v = line.split(" ", 1)
		word.append(w)
		vector.append( map(float, v.split()))
word_vec_dict = dict(zip(word,vector))	
print word_vec_dict['a']
#Load all processed sentences

file_name="word-descriptions.txt"
phrases=[]
with open(file_name, "r") as f:
	phrases = [line.rstrip('\n') for line in f]

for p in phrases:
	p= p.split()

#Add all word vectors in a phrase to form phrase vector


sum_vectors=[]

for p in xrange(len(phrases)):
	vec=[]
	for w in phrases[p]:
		if w in word_vec_dict:
			vec.append(word_vec_dict[w])
	if(vec==[]):
		print "no words in line in dict"
		print phrases[p]
	else:
		sum_vectors.append(np.sum(vec, axis=0))
		#sum_vectors[p]+=word_vec_dict[w]

#Output to word-vects.txt file
file_name="word-vects.txt"
print len(sum_vectors)
f=open(file_name, "w+")
for phrase in range(len(sum_vectors)):
		#print len(sum_vectors)
		#print range(len(sum_vectors))
		#print phrase
		#print sum_vectors[phrase]
		for n in sum_vectors[phrase]:
			#print n
			f.write(str(n))
			f.write(" ")
		f.write("\n")	





		
