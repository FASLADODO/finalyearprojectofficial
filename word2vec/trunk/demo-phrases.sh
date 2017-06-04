make
#if [ ! -e phrase-descriptions.txt ]; then
  sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < descriptions.txt | tr "A-Z'_.," "a-z " > phrase-descriptions.txt
#fi

time ./word2phrase -train phrase-descriptions.txt -output phrase0 -threshold 200 -debug 2
time ./word2phrase -train phrase0 -output phrase1 -threshold 150 -debug 2
time ./word2phrase -train phrase1 -output phrase2 -threshold 100 -debug 2
time ./word2phrase -train phrase2 -output phrase3 -threshold 50 -debug 2
time ./word2phrase -train phrase3 -output phrase4 -threshold 25 -debug 2
time ./word2phrase -train phrase4 -output phrase5 -threshold 10 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase0 -output news.2012.en.shuffled-norm0-phrase1 -threshold 100 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase1 -output news.2012.en.shuffled-norm0-phrase2 -threshold 50 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase2 -output news.2012.en.shuffled-norm0-phrase1 -threshold 25 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase1 -output news.2012.en.shuffled-norm0-phrase2 -threshold 10 -debug 2
#tr A-Z a-z < news.2012.en.shuffled-norm0-phrase2 > news.2012.en.shuffled-norm1-phrase2
time ./word2vec -train phrase1 -output vectors-phrase.txt -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 0 -threads 20 -binary 0 -iter 15 
time ./word2vec -train phrase1 -output vectors-phrase.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 0 -threads 20 -binary 1 -iter 15 -save-vocab phrase-vocab.txt
./distance vectors-phrase.bin

#word2phrase
#-train train file
#threshold, threshold to form phrases, higher means less phrases default 100
#reduces vocabulary by removing infrequent tokens
#uses out file to save resulting word vectors/clusters/phrases
##min-count discards words that appear less than <int> times, default is 5
#THRESHOLD REDUCES THE NUMBER OF WORDS, threshold smaller makes more words into phrases
