#!/bin/bash
# Run simulation: Experiment for every individual data set
# For every abnormal class
# We choose the abnormal class to be 0.
# The random seed is set to 2.
echo "Run Task 1"
cd ./Simulations_Experiments/
python train_Task1_fGAN_Simulation_Experiment.py
echo "Run Task 2"
python train_Task2_fGAN_Simulation_Experiment.py
echo "Run Task 3"
python train_Task3_fGAN_Simulation_Experiment.py
echo "Run Task 3 J"
python train_Task3_J_fGAN_Simulation_Experiment.py
exit 0
#for m in $(seq 0 2)
#do
#    echo "Seed: $m"
#    for i in $(seq 0 9)
#    do
#        echo "Running MNIST, Abnormal Digit: $i"
#        python train.py --dataset mnist --isize 32 --nc 1 --niter 15 --abnormal_class $i #--manualseed $m
#    done
#done
#exit 0
