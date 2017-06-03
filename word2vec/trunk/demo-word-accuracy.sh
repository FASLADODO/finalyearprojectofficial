make
if [ ! -e word-descriptions.txt ]; then
  sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < descriptions.txt | tr "A-Z,.\n'_" "a-z " > word-descriptions.txt
fi

time ./word2vec -train word-descriptions.txt -output vectors.bin -cbow 1 -size 200 -window 15 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15
#time ./word2vec -train word-descriptions.txt -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
./compute-accuracy vectors.bin 30000 < questions-words.txt
# to compute accuracy with the full vocabulary, use: ./compute-accuracy vectors.bin < questions-words.txt
