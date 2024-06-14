#!/bin/bash

WEEKS=52
CASES=("l2rpn_case14_sandbox_1x" "l2rpn_case14_sandbox_2x" "l2rpn_case14_sandbox_3x")
START=2019-01-05
N_SCENARIOS=1

for case in ${CASES[*]};
do
  echo "Generating data for case " $case
  chronix2grid --mode LRT \
		--output-folder `pwd`/../output_data \
		--input-folder `pwd`/../input_data \
		--ignore-warnings \
		--weeks $WEEKS \
		--case $case \
		--n_scenarios $N_SCENARIOS \
		--start-date $START \
		--by-n-weeks 4 \
		--seed-for-loads 936327420 --seed-for-res 936327420 --seed-for-dispatch 936327420 

done

