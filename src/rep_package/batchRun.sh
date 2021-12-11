#!/bin/bash

rm  "results/$1.txt"
touch  "results/$1.txt"
echo $2 $3 $4 $5 $6 $7 $8
for i in {1..10}
do
	python3 -W ignore main.py $i $2 $3 $4 $5 $6 $7 $8|egrep "para|systemSelector|fit|NRMSE|f1|elapsed|evaluating|encoder|decoder|bridge" |tee -a  "results/$1.txt"
done
