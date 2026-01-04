Conditional GAN for Shape Generation

This repository implements a Conditional Generative Adversarial Network (CGAN) that generates simple geometric shapes (circle, square, triangle) conditioned on class labels. By providing label information to both the generator and discriminator, the model enables controlled image generation.

Overview

A CGAN extends a standard GAN by conditioning the generation process on auxiliary information. In this project, discrete shape labels guide the generator to produce class-specific images at a resolution of 
64
×
64
64×64.

Architecture

The Generator takes a noise vector and a label embedding as input and produces a grayscale image using transposed convolutions, batch normalization, and ReLU activations, with Tanh at the output.

The Discriminator receives an image-label pair and predicts whether the image is real and label-consistent using strided convolutions and LeakyReLU activations.

Objective Function
min
⁡
𝐺
max
⁡
𝐷
𝑉
(
𝐷
,
𝐺
)
=
𝐸
𝑥
∼
𝑝
𝑑
𝑎
𝑡
𝑎
[
log
⁡
𝐷
(
𝑥
∣
𝑦
)
]
+
𝐸
𝑧
∼
𝑝
𝑧
[
log
⁡
(
1
−
𝐷
(
𝐺
(
𝑧
∣
𝑦
)
)
)
]
G
min
	​

D
max
	​

V(D,G)=E
x∼p
data
	​

	​

[logD(x∣y)]+E
z∼p
z
	​

	​

[log(1−D(G(z∣y)))]
Dataset

Synthetic grayscale images of size 
64
×
64
64×64 were used.
Classes: 0 – Circle, 1 – Square, 2 – Triangle
Images were normalized to 
[
−
1
,
1
]
[−1,1].

Training

Adam optimizer with learning rate 0.0002 and 
𝛽
1
=
0.5
β
1
	​

=0.5.
ReLU and Tanh were used in the generator, LeakyReLU in the discriminator.

Results

The model successfully generates shape-specific images when conditioned on labels. Initial mode collapse was observed and reduced through training adjustments. Generated shapes are visually distinct, confirming effective conditioning.

Usage
git clone https://github.com/your-username/conditional-gan-shapes.git
cd conditional-gan-shapes
pip install -r requirements.txt
python train.py
python generate.py --label square
