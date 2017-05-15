make
if [ ! -e word-descriptions.txt ]; then
  sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < descriptions.txt | tr "A-Z,.\n'_" "a-z " > word-descriptions.txt
fi

#sed -e is an editing intruction, used when mult instrs
#substitute a/b b for first inst a in line
#-c replace SET1 with its compliment/set2 (all not in SET1) tr SET1 SET2
# 0-9'_ \n
time ./word2vec -train word-descriptions.txt -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
./distance vectors.bin
