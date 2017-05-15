make
if [ ! -e word-descriptions.txt ]; then
  sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < descriptions.txt | tr "A-Z,.\n'_" "a-z " > word-descriptions.txt
fi
time ./word2vec -train word-descriptions.txt -output classes.txt -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -iter 15 -classes 500
sort classes.txt -k 2 -n > classes.sorted.txt
echo The word classes were saved to file classes.sorted.txt
