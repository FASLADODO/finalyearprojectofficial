make
sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < descriptions.txt | tr "A-Z'_.," "a-z " > word-descriptions.txt

#time ./word2phrase -train phrase-descriptions.txt -output phrase0 -threshold 200 -debug 2
#time ./word2phrase -train phrase0 -output phrase1 -threshold 150 -debug 2
#time ./word2phrase -train phrase1 -output phrase2 -threshold 100 -debug 2
#time ./word2phrase -train phrase2 -output phrase3 -threshold 50 -debug 2
#time ./word2phrase -train phrase3 -output phrase4 -threshold 25 -debug 2
#time ./word2phrase -train phrase4 -output phrase5 -threshold 10 -debug 2
#0 1e-6 1e-4 1e-2 1e-1
SAMPLES=(0)
#2 3 5 10 20
WINDOWS=( 5)
#1 2 3 5 7 10 15 20 50
NUM=`expr ${#SAMPLES[@]} - 1`
WIN=`expr ${#WINDOWS[@]} - 1`
for t in `seq 0 $NUM`; do
	for w in `seq 0 $WIN`; do
		printf "\n" 
            	echo Executing test with sample: ${SAMPLES[$t]} and window ${WINDOWS[$w]}
		time ./word2vec -train word-descriptions.txt -output vectors.bin -cbow 1 -size 200 -window ${WINDOWS[$w]} -negative 20 -hs 0 -sample ${SAMPLES[$t]} -threads 20 -binary 1 -iter 5
		./distance vectors.bin
		./compute-accuracy vectors.bin < questions-words2.txt

	done
done
