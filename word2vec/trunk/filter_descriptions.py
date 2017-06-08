import numpy as np
import glob
import os

file_in="phrase-descriptions/phrase-descriptions.txt"
file_out="phrase-descriptions/phrase-descriptions-corr.txt"

with open(file_in, "r") as f:
	with open(file_out, "wb+") as f_out:
		for line in f:
			words=line.split(" ")
			for word in words:
				if word in word_mappings:
					f_out.write(word_mappings[word])
					f_out.write(" ")
