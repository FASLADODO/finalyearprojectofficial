make
#if [ ! -e phrase-descriptions.txt ]; then
  sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < descriptions.txt | tr "A-Z'_.," "a-z " > phrase-descriptions.txt
#fi

time ./word2phrase -train phrase-descriptions.txt -output news.2012.en.shuffled-norm0-phrase0 -threshold 200 -debug 2
time ./word2phrase -train news.2012.en.shuffled-norm0-phrase0 -output news.2012.en.shuffled-norm0-phrase2 -threshold 100 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase0 -output news.2012.en.shuffled-norm0-phrase1 -threshold 100 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase1 -output news.2012.en.shuffled-norm0-phrase2 -threshold 50 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase2 -output news.2012.en.shuffled-norm0-phrase1 -threshold 25 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase1 -output news.2012.en.shuffled-norm0-phrase2 -threshold 10 -debug 2
tr A-Z a-z < news.2012.en.shuffled-norm0-phrase2 > news.2012.en.shuffled-norm1-phrase2
time ./word2vec -train news.2012.en.shuffled-norm1-phrase2 -output vectors-phrase.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 0 -threads 20 -binary 1 -iter 15
./compute-accuracy vectors-phrase.bin < questions-phrases.txt

#make
#if [ ! -e phrase-descriptions.txt ]; then
#  sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < descriptions.txt | tr "A-Z'_.," "a-z " > phrase-descriptions.txt
#fi
#time ./word2phrase -train phrase-descriptions.txt -output news.2012.en.shuffled-norm0-phrase0 -threshold 200 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase0 -output news.2012.en.shuffled-norm0-phrase1 -threshold 100 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase1 -output news.2012.en.shuffled-norm0-phrase2 -threshold 50 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase2 -output news.2012.en.shuffled-norm0-phrase1 -threshold 25 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase1 -output news.2012.en.shuffled-norm0-phrase2 -threshold 10 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase2 -output news.2012.en.shuffled-norm0-phrase1 -threshold 10 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase1 -output news.2012.en.shuffled-norm0-phrase2 -threshold 10 -debug 2
#tr A-Z a-z < news.2012.en.shuffled-norm0-phrase2 > news.2012.en.shuffled-norm1-phrase2
#time ./word2vec -train news.2012.en.shuffled-norm1-phrase2 -output vectors-phrase.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15

