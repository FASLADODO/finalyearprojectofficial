make
sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < descriptions.txt | tr "A-Z'_.," "a-z " > phrase-descriptions.txt

#time ./word2phrase -train phrase-descriptions.txt -output phrase0 -threshold 200 -debug 2
#time ./word2phrase -train phrase0 -output phrase1 -threshold 150 -debug 2
#time ./word2phrase -train phrase1 -output phrase2 -threshold 100 -debug 2
#time ./word2phrase -train phrase2 -output phrase3 -threshold 50 -debug 2
#time ./word2phrase -train phrase3 -output phrase4 -threshold 25 -debug 2
#time ./word2phrase -train phrase4 -output phrase5 -threshold 10 -debug 2
FILES=("phrase-descriptions.txt" "phrase0" "phrase1" "phrase2" "phrase3" "phrase4" "phrase5")
THRESHOLDS=(200 150 100 50 25 10)
VECTFILES=("vectors-phrase0.bin" "vectors-phrase1.bin" "vectors-phrase2.bin" "vectors-phrase3.bin" "vectors-phrase4.bin" "vectors-phrase5.bin")
#150 100 50 25 10
WINDOWS=( 2 3 5 )
#1 2 3 5 7 10 15 20 50
NUM=`expr ${#THRESHOLDS[@]} - 1`
WIN=`expr ${#WINDOWS[@]} - 1`
for t in `seq 0 $NUM`; do
	for w in `seq 0 $WIN`; do
		printf "\n" 
            	echo Executing test with threshold: ${THRESHOLDS[$t]} and window ${WINDOWS[$w]} and filein  ${FILES[$t]} and fileout ${FILES[$t+1]}
		FILEOUT="vectors-phrase-win"${WINDOWS[$w]}"-threshold"${THRESHOLDS[$t]}".bin"
		FILEOUT2="vectors-phrase-win"${WINDOWS[$w]}"-threshold"${THRESHOLDS[$t]}".txt"
		time ./word2phrase -train ${FILES[$t]} -output ${FILES[$t+1]} -threshold ${THRESHOLDS[$t]} -debug 2
		time ./word2vec -train ${FILES[$t+1]} -output $FILEOUT -cbow 1 -size 200 -window ${WINDOWS[$w]} -negative 25 -hs 0 -sample 0 -threads 20 -binary 1 -iter 15 
		time ./word2vec -train ${FILES[$t+1]} -output $FILEOUT2 -cbow 1 -size 200 -window ${WINDOWS[$w]} -negative 25 -hs 0 -sample 0 -threads 20 -binary 0 -iter 15
	        ./compute-accuracy $FILEOUT < questions-phrases.txt
	done
done

#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase0 -output news.2012.en.shuffled-norm0-phrase1 -threshold 100 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase1 -output news.2012.en.shuffled-norm0-phrase2 -threshold 50 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase2 -output news.2012.en.shuffled-norm0-phrase1 -threshold 25 -debug 2
#time ./word2phrase -train news.2012.en.shuffled-norm0-phrase1 -output news.2012.en.shuffled-norm0-phrase2 -threshold 10 -debug 2
#tr A-Z a-z < news.2012.en.shuffled-norm0-phrase2 > news.2012.en.shuffled-norm1-phrase2


#time ./word2vec -train phrase1 -output vectors-phrase.txt -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 0 -threads 20 -binary 0 -iter 15 

#./distance vectors-phrase.bin
