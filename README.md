# OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary
OoD Minimum Anomaly Score GAN (OMASGAN)

Code Repository for 'OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary'

Abstract of Paper:
Deep generative models trained in an unsupervised manner encounter the problem of setting high likelihood and low reconstruction loss to Out-of-Distribution (OoD) samples. This increases the Type II errors (false negatives, misses of anomalies) and decreases the Anomaly Detection (AD) performance. Also, deep generative models for AD suffer from the rarity of anomalies. To address these limitations, we propose a new model to perform active negative sampling and training. Our OoD Minimum Anomaly Score GAN (OMASGAN) uses data only from the normal class. Our model generates samples on the boundary of the support of the data distribution. These boundary samples are abnormal data and we perform retraining for AD using normal and the generated minimum-anomaly-score OoD samples. We can use any f-divergence distribution metric. Likelihood and invertibility are not needed. We use a discriminator for inference and the evaluation of OMASGAN on images using the leave-one-out methodology shows that it outperforms state-of-the-art benchmarks. Using the Area Under the Receiver Operating Characteristics curve (AUROC), OMASGAN yields an improvement of at least 0.24 and 0.07 points on average on MNIST and CIFAR-10 respectively over recent benchmarks.

Flowchart Diagram:

![plot](./images/FlowchartOMASGAN.png)

Diagram of OMASGAN:

![plot](./images/Flowchart_OMASGAN.png)

Algorithm of OMASGAN:

![plot](./images/Illustration_OMASGAN.png)

This Code Repository contains a PyTorch implementation for the OMASGAN model.

As shown in the flowchart diagram, OMASGAN performs model retraining by including negative samples, where the negative samples are generate by our negative data augmentation methodology.

Our negative data augmentation methodology: OMASGAN generates minimum anomaly score OoD samples around the data using a strictly decreasing function of a distribution metric between the boundary samples and the data.

Environments - Requirements: Python 3.7 and PyTorch 1.3

Date: Wednesday 13 January 2021: Creation of the Code Repository for OMASGAN.

Future Date: Saturday 8 May 2021: Author Notification: Make the Code Repository non-anonymous, release the code, and make the code public.

Date: Monday 18 January 2021: Creation of the Folder Proof Of Concept Experiment for the Boundary Task and the Retraining Task of OMASGAN.

The boundary algorithm (Task 2) and the retraining function (Task 3) of OMASGAN are new.

In the Proof Of Concept Experiment for the Boundary and Retraining Tasks, according to Table 4 of the f-GAN paper, we use the Pearson Chi-Squared f-divergence distribution metric and we note that after Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.

Example usage: "cd ./proof_of_concept_experiment/" and then "python train_Task1_fGAN_Proof_of_Concept.py", "python -m train_Task1_fGAN_Proof_of_Concept", "python Task1_MNIST_fGAN_Proof_of_Concept.py", or "python Task1_MNIST2_fGAN_Proof_of_Concept.py"

Also: Example usage: "cd ./proof_of_concept_experiment/Task1_CIFAR10_MNIST_KLWGAN_Proof_of_Concept/" and then "python train_Task1_KLWGAN_Proof_of_Concept.py --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5"

Project Website: [OMASGAN Project](https://anonymous.4open.science/r/2c122800-a538-4357-b452-a8d0e9a92bee/).

Acknowledgement: Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)

Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)

Additional acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GAN-Models](https://github.com/shayneobrien/generative-models), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs)

This website is best viewed in Chrome or Firefox.
