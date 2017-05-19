import numpy as np
import glob
import os
NONE=0
COLOR=1

WEIGHT_MODE=NONE
NORMALISE=1


#Load all word vector associations
#FILEOUT2="vectors-phrase-win"${WINDOWS[$w]}"-threshold"${THRESHOLDS[$t]}".txt"
#print glob.glob("/home/rs6713/Documents/finalyearprojectofficial/word2vec/trunk/vectors-phrase-win*.txt")
files=["phrasevectors-txt/"+i for i in os.listdir("/home/rs6713/Documents/finalyearprojectofficial/word2vec/trunk/phrasevectors-txt") if i.find("vectors-phrase-win")!=-1 and i.find(".txt")!=-1]
thresh_dict={"200": "phrase-descriptions/phrase0","150": "phrase-descriptions/phrase1", "100" : "phrase-descriptions/phrase2", "50": "phrase-descriptions/phrase3", "25": "phrase-descriptions/phrase4", "10": "phrase-descriptions/phrase5", "0": "phrase-descriptions/phrase-descriptions.txt"}
print files
#phrase_files= ["phrase0", "phrase1", "phrase2", "phrase3", "phrase4", "phrase5"]
#phraseout_files= ["phrase0-vects.txt", "phrase1-vects.txt", "phrase2-vects.txt", "phrase3-vects.txt", "phrase4-vects.txt", "phrase5-vects.txt"]
for file_in in files:
	phraseout_file="out"+file_in.replace('phrasevectors-txt/', '')
	p,m=file_in.split("-threshold")
	phrasefileindex,m=m.split(".")#threshold maps
        print phrasefileindex + " " + thresh_dict[phrasefileindex]  	

	#file_name="vectors-phrase.txt"
	word=[] #Contains key of words
	vector=[] #Contains 
	with open(file_in, "r") as f:
		header=f.readline()
	        print header
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
	phrases=[]
	with open(thresh_dict[phrasefileindex], "r") as f:
		phrases = [line.rstrip('\n') for line in f]

	for p in phrases:
		p= p.split()


	#Add all word vectors in a phrase to form sentence vector
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
			#Sums word vectors to sentence vector, maybe normalised
			if(NORMALISE):
				length=len(vec)
				normal_vec=np.sum(vec,axis=0)/length
				sum_vectors.append(normal_vec)
			else:
				sum_vectors.append(np.sum(vec, axis=0))
			

	#Output to word-vects.txt file

	print len(sum_vectors)
	f=open("matlab_sentence_vectors/"+"mode"+str(WEIGHT_MODE)+"_norm"+str(NORMALISE)+phraseout_file.replace("-","_"), "w+")
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
