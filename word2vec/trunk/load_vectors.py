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
words=load_bin_vec("vectors.bin")
with open("out-words.txt", "w+") as f:
	for line in words:
		f.write(line)
		f.write("\n")
		

