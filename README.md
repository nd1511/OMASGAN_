# OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary
Out-of-Distribution Minimum Anomaly Score GAN (OMASGAN)

Code Repository for 'OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary'

## Abstract of Paper:
Deep generative models trained in an unsupervised manner encounter the problem of setting high likelihood, high probability density, and low reconstruction loss to Out-of-Distribution (OoD) samples. This increases the Type II errors (false negatives, misses of anomalies) and decreases the Anomaly Detection (AD) performance. Also, deep generative models for AD suffer from the rarity of anomalies problem. To address these limitations, we propose the new OoD Minimum Anomaly Score GAN (OMASGAN) model. OMASGAN addresses the rarity of anomalies problem by generating strong abnormal samples on the boundary of the support of the data distribution, using data only from the normal class. OMASGAN improves the AD performance by retraining including the abnormal minimum-anomaly-score OoD samples generated by our negative sampling augmentation methodology. OMASGAN uses any f-divergence distribution metric in its variational representation, and explicit likelihood and invertibility are not needed. The proposed model uses a discriminator for inference and the evaluation of OMASGAN on image data using the leave-one-out methodology shows that it achieves an improvement of at least 0.24 and 0.07 points in AUROC on average on MNIST and CIFAR-10 data, respectively, over recently proposed state-of-the-art AD benchmarks.

## Flowchart Diagram:

![plot](./Figures_Images/FlowchartOMASGAN.png)

Figure 1: Flowchart of the OMASGAN model for AD which generates minimum-anomaly-score OoD samples on the boundary of the support of the data distribution and subsequently uses these generated boundary samples to train a discriminative model for detecting abnormal data.

![plot](./Figures_Images/Flowchart_OMASGAN.png)

Figure 2: Training of the OMASGAN model for AD in images using active negative sampling and training by generating strong abnormal OoD samples on the boundary of the data distribution.

![plot](./Figures_Images/Illustration_OMASGAN.png)

Figure 3: Illustration of the OMASGAN algorithm for AD where **x**~p<sub>**x**</sub>, G(**z**)~p<sub>g</sub>, and G'(**z**)~p<sub>g'</sub>. The figure shows the OMASGAN Tasks and the data distribution, p<sub>**x**</sub>, the data model distribution, p<sub>g</sub>, the data model distribution after retraining, p<sub>g'</sub>, and the samples from the boundary of the support of the data distribution, B(**z**)~p<sub>b</sub>.

## Usage:

For the evaluation of the proposed OMASGAN model, we use the leave-one-out (LOO) evaluation methodology and the image data sets [MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). We also use synthetic data for the evaluation of the OMASGAN model and all the evaluation results can be found in the paper "OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary".

Simulations Experiments folder: The Tasks of OMASGAN, including the Boundary Task and the Retraining Task. In the boundary algorithm (Task 2), the boundary model is trained to perform sample generation on the boundary of the data distribution by starting from within the data distribution (Task 1). In the retraining function (Task 3), as shown in the flowchart diagram in Figure 1, OMASGAN performs model retraining for AD by including negative samples, where the negative samples are generated by our negative data augmentation methodology. Regarding our negative data augmentation methodology, OMASGAN generates minimum anomaly score OoD samples around the data using a strictly decreasing function of a distribution metric between the boundary samples and the data.

In the Simulations Experiments folder, for the Boundary and Retraining Tasks, according to Table 4 of the f-GAN paper, we use the Pearson Chi-Squared f-divergence distribution metric and we note that after Pearson Chi-Squared, the next best are KL and then Jensen-Shannon.

To run f-GAN-based OMASGAN training using the LOO methodology on MNIST data, for abnormal_class_LOO (train_Task1_fGAN_Simulation_Experiment.py), run the bash script:
```
cd ./Experiments/
sh run_OMASGAN_fGAN_MNIST.sh
```

Example usage:
```
cd ./Simulations_Experiments/
python train_Task1_fGAN_Simulation_Experiment.py
#python -m train_Task1_fGAN_Simulation_Experiment
python Task1_MNIST_fGAN_Simulation_Experiment.py
python Task1_MNIST2_fGAN_Simulation_Experiment.py
```

Also, example usage:
```
cd ./Simulations_Experiments/Task1_CIFAR10_MNIST_KLWGAN_Simulation_Experiment/
python train_Task1_KLWGAN_Simulation_Experiment.py --abnormal_class 0 --shuffle --batch_size 64 --parallel --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --dataset C10 --data_root ./data/ --G_ortho 0.0 --G_attn 0 --D_attn 0 --G_init N02 --D_init N02 --ema --use_ema --ema_start 1000 --start_eval 50 --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --loss_type kl_5 --seed 2 --which_best FID --model BigGAN --experiment_name C10Ukl5
```

The use of torch.nn.DataParallel(model) is recommended along with the use of torch.save(model.module.state_dict(), "./.pt") instead of torch.save(model.state_dict(), "./.pt"). Also, saving the best model is recommended by using "best_loss = float('inf')" and "if loss.item()<best_loss: best_loss=loss.item(); torch.save(model.module.state_dict(), "./.pt")".

After saving the trained model from Task 1: Example usage:
```
cd ./Simulations_Experiments/
python train_Task2_fGAN_Simulation_Experiment.py
```

Then, after saving the trained models from Tasks 1 and 2: Example usage:
```
cd ./Simulations_Experiments/
python train_Task3_fGAN_Simulation_Experiment.py
```

Next, after saving the trained models from Tasks 1, 2, and 3: Example usage:
```
cd ./Simulations_Experiments/
python train_Task3_J_fGAN_Simulation_Experiment.py
```

For multimodal distributions with disconnected components, for synthetic data, OMASGAN successfully forms the boundary of the support of the data distribution in [OMASGAN-Task2](https://github.com/Anonymous-Author-2021/OMASGAN/blob/main/Simulations_Experiments/Toy_Data_Simulation_Experiment/Toy_Data_Image1_Task2.pdf), [Task2-Boundary](https://github.com/Anonymous-Author-2021/OMASGAN/blob/main/Simulations_Experiments/Toy_Data_Simulation_Experiment/Toy_Data_Image11_Task2.pdf), [Boundary-Formation](https://github.com/Anonymous-Author-2021/OMASGAN/blob/main/Simulations_Experiments/Toy_Data_Simulation_Experiment/Toy_Data_Image14_Task2.pdf), and [OMASGAN-AUROC](https://github.com/Anonymous-Author-2021/OMASGAN/blob/main/Simulations_Experiments/Toy_Data_Simulation_Experiment/Toy_Data_Image19_AUROC.pdf).

For synthetic data, example usage:
```
cd ./Simulations_Experiments/Toy_Data_Simulation_Experiment/
python train_Toy_Data_fGAN_Simulation_Experiment.py
```

## Further Usage Information:

To run the code, we use a virtual environment and conda. For the versions of the libraries we use, see the requirements.txt file which has been created by using "pip freeze > requirements.txt". For installing the versions of the libraries we use, run "pip install -r requirements.txt".

To clone the Code Repository, run:
```
git clone https://github.com/Anonymous-Author-2021/OMASGAN.git
```

This Code Repository contains a PyTorch implementation for the OMASGAN model.

Environments - Requirements: Python 3.7 and PyTorch 1.2

Future Date: Saturday 8 May 2021: Author Notification: Make the Code Repository non-anonymous, release the source code, and make the source code public.

Project Website: [OMASGAN Project](https://anonymous.4open.science/r/2c122800-a538-4357-b452-a8d0e9a92bee/).

For evaluation, we use [MNIST](http://yann.lecun.com/exdb/mnist/), [Fashion-MNIST](https://www.kaggle.com/zalando-research/fashionmnist), [KMNIST](https://github.com/rois-codh/kmnist), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [SVHN](http://ufldl.stanford.edu/housenumbers/), [CINIC-10](https://www.kaggle.com/mengcius/cinic10), [STL-10](https://cs.stanford.edu/~acoates/stl10/), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and data from the Outlier Detection Data Sets (ODDS) repository ([ODDS](http://odds.cs.stonybrook.edu/)).

This website is best viewed in Chrome or Firefox.

## Acknowledgements:

Thanks to the repositories: [PyTorch-Template](https://github.com/victoresque/pytorch-template "PyTorch Template"), [Generative Models](https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py), [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan), and [KLWGAN](https://github.com/ermongroup/f-wgan/tree/master/image_generation).

Acknowledgement: Thanks to the repositories: [f-GAN](https://github.com/nowozin/mlss2018-madrid-gan/blob/master/GAN%20-%20CIFAR.ipynb), [GANs](https://github.com/shayneobrien/generative-models), [Boundary-GAN](https://github.com/wiseodd/generative-models/blob/master/GAN/boundary_seeking_gan/bgan_pytorch.py), [fGAN](https://github.com/wiseodd/generative-models/blob/master/GAN/f_gan/f_gan_pytorch.py), and [Rumi-GAN](https://github.com/DarthSid95/RumiGANs).

Also, thanks to the repositories: [Negative-Data-Augmentation](https://anonymous.4open.science/r/99219ca9-ff6a-49e5-a525-c954080de8a7/), [Negative-Data-Augmentation-Paper](https://openreview.net/forum?id=Ovp8dvB8IBH), and [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).

Additional acknowledgement: Thanks to the repositories: [Pearson-Chi-Squared](https://anonymous.4open.science/repository/99219ca9-ff6a-49e5-a525-c954080de8a7/losses.py), [DeepSAD](https://github.com/lukasruff/Deep-SAD-PyTorch), and [GANomaly](https://github.com/samet-akcay/ganomaly).

All the acknowledgements, references, and citations can be found in the paper "OMASGAN: Out-of-Distribution Minimum Anomaly Score GAN for Sample Generation on the Boundary".

## References:
Asokan, S. and Seelamantula, C., “Teaching a GAN What Not to Learn,” in Proceedings 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds), Vancouver, Canada, December 2020.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y., “Generative Adversarial Nets,” in Proceedings Advances in Neural Information Processing Systems (NIPS), pp. 2672–2680, Montréal, Canada, December 2014.

Nalisnick, E.,  Matsukawa, A., Teh, Y., Gorur, D., and Lakshminarayanan, B., “Do Deep Generative Models Know What They Don’t Know?,” in Proceedings International Conference on Learning Representations (ICLR), New Orleans, USA, May 2019.

Nowozin, S., Cseke, B., and Tomioka, R., “f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization,” in Proceedings Thirtieth Conference on Neural Information Processing Systems (NIPS), Barcelona, Spain, December 2016.

Song, J. and Ermon, S., “Bridging the Gap Between f-GANs and Wasserstein GANs,” in Proceedings International Conference on Machine Learning (ICML), pp. 9078-9087, vol. 119, Daumé III, H. and Singh, A. (eds), July 2020.

Zaheer, M., Lee, J., Astrid, M., and Lee, S., “Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm,” in Proceedings IEEE/CVF Conference Computer Vision and Pattern Recognition (CVPR), pp. 14171-14181, Seattle, Washington, USA, June 2020. DOI: 10.1109/CVPR42600.2020.01419
