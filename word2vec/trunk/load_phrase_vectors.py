import numpy as np
import glob
import os

#Load all word vector associations
#FILEOUT2="vectors-phrase-win"${WINDOWS[$w]}"-threshold"${THRESHOLDS[$t]}".txt"
#print glob.glob("/home/rs6713/Documents/finalyearprojectofficial/word2vec/trunk/vectors-phrase-win*.txt")
files=[i for i in os.listdir("/home/rs6713/Documents/finalyearprojectofficial/word2vec/trunk") if i.find("vectors-phrase-win")!=-1 and i.find(".txt")!=-1]
print files
phrase_files= ["phrase0", "phrase1", "phrase2", "phrase3", "phrase4", "phrase5"]
#phraseout_files= ["phrase0-vects.txt", "phrase1-vects.txt", "phrase2-vects.txt", "phrase3-vects.txt", "phrase4-vects.txt", "phrase5-vects.txt"]
for file_in in files:
	phraseout_file="out"+file_in
	p,m=file_in.split("-threshold")
	m,phrasefileindex=p.split("-win")


	file_name="vectors-phrase.txt"
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

#file_name="phrase2"#COuld be phrase0, phrase1
#for i in range(len(phrase_files)):
	phrases=[]
	with open(phrase_files[i], "r") as f:
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
	file_name="phrase-vects.txt"
	print len(sum_vectors)
	f=open(phraseout_files[i], "w+")
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
