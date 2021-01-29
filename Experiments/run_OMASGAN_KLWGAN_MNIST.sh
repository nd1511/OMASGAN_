#!/bin/bash
# OMASGAN: Bash script
# Simulation: Run experiment for every individual image data set
for i in $(seq 0 9)
do
  # Manual Seed: $m
  for m in $(seq 0 2)
  do
    # Seed: $m
    echo "MNIST: Run OMASGAN Task 1"
    cd ../Simulations_Experiments/Task1_CIFAR10_MNIST_KLWGAN_Simulation_Experiment/
    python train_Task1_KLWGAN_Simulation_Experiment.py --select_dataset mnist --abnormal_class 0 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
    echo "MNIST: Run Task 2"
    cd ../Task2_CIFAR10_MNIST_KLWGAN_Simulation_Experiment/
    #python train.py --select_dataset mnist --abnormal_class 0 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
    python train.py --select_dataset mnist --abnormal_class 0 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ../Task1_CIFAR_MNIST_KLWGAN_Simulation_Experiment/data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
    echo "MNIST: Run Task 3"
    cd ../Task3_CIFAR10_MNIST_KLWGAN_Simulation_Experiment/
    python train.py --select_dataset mnist --abnormal_class 0 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ../Task1_CIFAR_MNIST_KLWGAN_Simulation_Experiment/data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
    echo "MNIST: Run Task 3 J"
    cd ../Task3_J_CIFAR10_MNIST_KLWGAN_Simulation_Experiment/
    python train.py --select_dataset mnist --abnormal_class 0 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ../Task1_CIFAR_MNIST_KLWGAN_Simulation_Experiment/data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
  done
done
exit 0
