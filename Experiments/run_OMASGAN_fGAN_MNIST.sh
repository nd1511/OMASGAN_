#!/bin/bash
# Run simulation: Experiment for every individual image data set
# For every abnormal class
# i is the abnormal OoD leave-out class
for i in $(seq 0 9)
do
  # We first choose the abnormal class to be 0.
  # The random seed is set to 2.
  echo "Run OMASGAN Task 1"
  cd ../Simulations_Experiments/
  python train_Task1_fGAN_Simulation_Experiment.py --abnormal_class $i
  echo "Run Task 2"
  python train_Task2_fGAN_Simulation_Experiment.py --abnormal_class $i
  echo "Run Task 3"
  python train_Task3_fGAN_Simulation_Experiment.py --abnormal_class $i
  echo "Run Task 3 J"
  python train_Task3_J_fGAN_Simulation_Experiment.py --abnormal_class $i
  # The final step is to perform AD and detect anomalies.
  # Inference: Use the functions in "test_Evaluation_Metrics_Simulation_Experiment.py".
  #python test_Evaluation_Metrics_Simulation_Experiment.py
done
# End the bash script
exit 0
