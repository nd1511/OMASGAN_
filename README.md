# OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary
OoD Minimum Anomaly Score GAN (OMASGAN)

Code Repository for 'OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary'

Abstract of Paper:
Deep generative models trained in an unsupervised way encounter the problem of setting high likelihood and low reconstruction loss to Out-of-Distribution (OoD) samples.
This increases the misses of abnormal data and decreases the Anomaly Detection (AD) performance.
Also, deep generative models for AD suffer from the rarity of anomalous events.
To address these limitations, we propose a model to perform active negative sampling and training.
Our OoD Minimum Anomaly Score GAN (OMASGAN) performs self-supervised learning using data only from the normal class.
Our f-divergence-based GAN generates samples on the boundary of the support of the data distribution.
These boundary samples are strong abnormal data and we perform retraining for AD using normal and the generated minimum-anomaly-score OoD samples.
We use any f-divergence distribution metric and a discriminator, and likelihood and/or invertibility are not needed.
The evaluation of the proposed OMASGAN model on image data using the leave-one-out methodology shows that it outperforms several state-of-the-art benchmarks.
Using the Area Under the Receiver Operating Characteristics curve (AUROC), OMASGAN yields an improvement of at least 0.24 and 0.07 points on average on MNIST and CIFAR-10 respectively over recent state-of-the-art AD benchmark models.

Flowchart Diagram:

![plot](./images/Diagram_OMASGAN.png)

Diagram of OMASGAN:

![plot](./images/Flowchart_OMASGAN.png)

Algorithm of OMASGAN:

![plot](./images/Illustration_OMASGAN.png)

This Code Repository contains a PyTorch implementation of the OMASGAN model.

Environments - Requirements: Python 3.7 and PyTorch 1.3

Date: Wednesday 13 January 2021: Creation of the Code Repository for OMASGAN.

Future Date: Saturday 8 May 2021: Author Notification: Make the Code Repository non-anonymous, make the code public, and release the code.

Project Website: [OMASGAN Project](https://anonymous.4open.science/r/2c122800-a538-4357-b452-a8d0e9a92bee/).

Acknowledgement: Thanks to the repositories: [PyTorch Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation)

Also, thanks to the repositories: [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan)

This website is best viewed in Chrome or Firefox.
