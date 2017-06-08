import numpy as np
import glob
import os
NONE=0
SIMWORDS=1
USELESSWORDS=2
LOWFREQWORDS=3


sim_mappings={'brownish': 'brown','all-black':'black', 'elderly': 'old', 'grayed':'grey', 'trainers':'sneakers', '20':'twenties',  "rucksack":"backpack", "tshirt":"t-shirt", "-":"","t_shirt":"t-shirt", "ruck_sack":"backpack","back_pack":"back_pack","wristwatch":"watch", "tee":"t-shirt", "flabby":"large", "plumpy":"large", "fat":"large", "mobile":"phone", "hand_bag":"handbag", "sport":"sports", "cellphone":"phone", "mobilephone":"phone"}
useless_words=["a","the","is","with","of","probably","definitely","perfectly","almost","super", "appears", "this", "an", "that", "seen", "at", "as", "s", "also", "while", "early", "may", "either","see", "but", "there", "well", "so","here", "what","we","by"];
WEIGHT_MODE=USELESSWORDS
NORMALISE=3#1 IS MEAN, 2 IS MAX, 0 is matrix form

fileFrequencies="word-vocab.txt"

#Load all word vector associations
#FILEOUT2="vectors-phrase-win"${WINDOWS[$w]}"-threshold"${THRESHOLDS[$t]}".txt"
#print glob.glob("/home/rs6713/Documents/finalyearprojectofficial/word2vec/trunk/vectors-phrase-win*.txt")                                  vectors-phrase-win
files=["phrasevectors-txt/"+i for i in os.listdir("/home/rs6713/Documents/finalyearprojectofficial/word2vec/trunk/phrasevectors-txt") if i.find(".txt")!=-1]
#i.find("vectors-phrase-win3-threshold100")!=-1 and , how to pick specific ones
thresh_dict={"200": "phrase-descriptions/phrase0","150": "phrase-descriptions/phrase1", "100" : "phrase-descriptions/phrase2", "50": "phrase-descriptions/phrase3", "25": "phrase-descriptions/phrase4", "10": "phrase-descriptions/phrase5", "0": "phrase-descriptions/phrase-descriptions.txt"}
print files
#phrase_files= ["phrase0", "phrase1", "phrase2", "phrase3", "phrase4", "phrase5"]
#phraseout_files= ["phrase0-vects.txt", "phrase1-vects.txt", "phrase2-vects.txt", "phrase3-vects.txt", "phrase4-vects.txt", "phrase5-vects.txt"]
wordFreqDict={}
with open(fileFrequencies, "r") as f:
	for line in f:
		w,freq=line.split(" ")
                freq=freq.replace("\n","")
		wordFreqDict[w] = freq
print wordFreqDict
for file_in in files:
	phraseout_file="out"+file_in.replace('phrasevectors-txt/', '')
	p,m=file_in.split("-threshold")
	phrasefileindex,m=m.split("-size")#threshold maps
	#size=int( m.replace(c,'')
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
	#print word_vec_dict['a']

	#Load all processed sentences
	phrases=[]
	with open(thresh_dict[phrasefileindex], "r") as f:
		phrases = [line.rstrip('\n') for line in f]

	for p in phrases:
		p= p.split()


	#Add all word vectors in a phrase to form sentence vector
	#This can be done via mean, max, or matrix formation
	
	length_vectors=np.array([])

	for p in xrange(len(phrases)):
		vec=[]#vec is array of word vectors that form a phrase
		for w in phrases[p]:
			temp=w
			if(WEIGHT_MODE==SIMWORDS):
				if w in sim_mappings:
					temp=sim_mappings[w]
			if(WEIGHT_MODE==USELESSWORDS):
				if w in useless_words:
					temp=""
			if(WEIGHT_MODE==LOWFREQWORDS):
				word_freq=wordFreqDict[w]
 				if(int(word_freq)<20):
					temp=""
			if temp in word_vec_dict:
				vec.append(word_vec_dict[temp])
		if(vec!=[]):
			#print len(phrases[p]), counts chars
			
			length_vectors=np.append(length_vectors, len(vec))

	mean_length=int(np.mean(length_vectors))
	print "the mean length of %d sentences is:  %d" %(len(phrases), np.mean(length_vectors)) 
	if(NORMALISE==1 or NORMALISE==2):
		sum_vectors=np.empty([len(phrases), len(word_vec_dict['a'])])
	else:
		sum_vectors=np.empty([len(phrases),mean_length, len(word_vec_dict['a'])])

	for p in xrange(len(phrases)):
		vec=[]#vec is array of word vectors that form a phrase
		for w in phrases[p]:
			
			if w in word_vec_dict:
				vec.append(word_vec_dict[w])
				
		if(vec==[]):
			print "no words in line in dict"
			print phrases[p]
		else:
			vec=np.array(vec);
			#Sums word vectors to sentence vector, maybe normalised
			if(NORMALISE==1):
				length=vec.shape[0]
				normal_vec=np.sum(vec,axis=0)/length
				sum_vectors[p]=normal_vec
			elif(NORMALISE==2):
				sum_vectors[p]=np.amax(vec,axis=0)
			
			else:
				#print vec
				#print "mean length %d" %mean_length
				#print vec.shape[0]
				#print len(vec)
				if(vec.shape[0]==mean_length):
					sum_vectors[p]=vec
				elif(vec.shape[0]>=mean_length):
					sum_vectors[p]=vec[0:mean_length]
					#sum_vectors=np.append(sum_vectors,vec[0:mean_length-1], axis=0)
				else:
					#print "length difference %d" % (mean_length-vec.shape[0])
					zero_vec= np.array([[0 for col in range(vector_size)] for row in range( int(mean_length-len(vec)))])
					#print "zero_vec size %d %d" % (len(zero_vec), len(zero_vec[0]))
					#print "vec size %d %d" % (vec.shape[0], vec[0].shape[0])
					vec=np.append(vec, zero_vec, axis=0)
					#print "vec new size %d %d" % (len(vec), len(vec[1]))
					sum_vectors[p]=vec
		
	print "The shape of sum_vectors is %s" % str(np.shape(sum_vectors))

	#sum_vecotrs is vector of all phrase vectors
	#Output to word-vects.txt file
	
	
	f=open("matlab_sentence_vectors/"+"mode"+str(WEIGHT_MODE)+"_norm"+str(NORMALISE)+phraseout_file.replace("-","_"), "w+")
	for phrase in range(len(sum_vectors)):
			#print len(sum_vectors)
			#print range(len(sum_vectors))
			#print phrase
			#print sum_vectors[phrase]
		if(NORMALISE==1 or NORMALISE==2):

			for n in xrange(np.shape(sum_vectors[phrase])[0]):
				#print n
				f.write(str(sum_vectors[phrase][n]))
				f.write(" ")
			f.write("\n")	
		else:
			for n in xrange(np.shape(sum_vectors[phrase])[0]):
				for i in xrange(np.shape(sum_vectors[phrase][n])[0]):
					f.write(str(sum_vectors[phrase][n][i]))
					f.write(" ")
				f.write(", ")
			f.write("\n")

