#!/bin/bash

TEMPLATE="main.template"
OUT="main.c"
BLOCK_SIZES=(1 2 4 8 10 20)
TRIALS=3
REGEX="The kernel took (.+\..*) ms"
PROG="./matmul"

for block_size in ${BLOCK_SIZES[@]}
do
  sed "s/{{ BLOCK_SIZE }}/${block_size}/" $TEMPLATE > $OUT
  make

  avg=0
  for i in `seq $TRIALS`
  do
    if [[ `$PROG` =~ $REGEX ]]
    then
      dur=${BASH_REMATCH[1]}
      avg=`echo "$avg + $dur" | bc -l` 
    fi
  done
  avg=`echo "$avg / $TRIALS" | bc -l`

  echo "BLOCK_SIZE=$block_size took $avg ms"
done
