# Clustered VAE
<p class="lead">BSD 3-Clause License

Copyright (c) 2017, Yida Wang
All rights reserved.
</p>

VAE with clustered latent space designed by Yida Wang.
There is also a notebook useful for understanding the basic VAE definition based on Tensorflow through the notebook. I personably ever took a deep learning course on Kadenze CADL repo on Github.

## Clustering methods

Firstly, you should make 2 directories ```test_sita_vae``` and ```test_sita_clvae``` for saving results and figures. I use K-Means and GMM for clustering on latent space.

## Tensorflow Install

There might be several problems if there are multiple GPUs

1. cudnn problem: [cudnn](https://developer.nvidia.com/rdp/cudnn-download) for specific version. I use all compiled headers and libraries for installation according to CUDA version and platform. There is a useful discussion about [How can I install CuDNN on Ubuntu 16.04](https://askubuntu.com/questions/767269/how-can-i-install-cudnn-on-ubuntu-16-04).
It might be something like this:
```sh
cd folder/extracted/contents
sudo cp -P include/cudnn.h /usr/include
sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
```
Those steps works on my Ubuntu 16.04 station with 4 Nvidia 1080 GPUs

We can have a check on the CUDA NVCC tools:
```sh
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2015 NVIDIA Corporation
Built on Tue_Aug_11_14:27:32_CDT_2015
Cuda compilation tools, release 7.5, V7.5.17
```

2. Volatile GPU-Util problem: it might be all 0%, so you could have a try on Celeb video using them seperately by:
```sh
## Four tests on MNIST databse with variational/clustered without convolutional
screen -r ae_mnist
CUDA_VISIBLE_DEVICES=0 python3 test_mnist.py -o result_mnist_ae
screen -d

screen -r vae_mnist
CUDA_VISIBLE_DEVICES=1 python3 test_mnist.py -v -o result_mnist_vae
screen -d

screen -r clae_mnist
CUDA_VISIBLE_DEVICES=2 python3 test_mnist.py -k  -o result_mnist_clae
screen -d

screen -r clvae_mnist
CUDA_VISIBLE_DEVICES=3 python3 test_mnist.py -v -k -o result_mnist_clvae
screen -d

## Four tests on ShapeNet databse with variational/clustered with convolutional
screen -r ae_shapenet
CUDA_VISIBLE_DEVICES=0 python3 test_shapenet.py -c -o result_shapenet_ae
screen -d

screen -r vae_shapenet
CUDA_VISIBLE_DEVICES=1 python3 test_shapenet.py -c -v -o result_shapenet_vae
screen -d

screen -r clae_shapenet
CUDA_VISIBLE_DEVICES=2 python3 test_shapenet.py -c -k -o result_shapenet_clae
screen -d

screen -r clvae_shapenet
CUDA_VISIBLE_DEVICES=3 python3 test_shapenet.py -c -v -k -o result_shapenet_clvae
screen -d
```
It will be something like this:

```sh
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 0000:03:00.0      On |                  N/A |
| 46%   80C    P2   176W / 250W |  10903MiB / 11170MiB |     83%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 0000:04:00.0     Off |                  N/A |
| 38%   68C    P2   135W / 250W |  10547MiB / 11172MiB |     86%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 108...  Off  | 0000:81:00.0     Off |                  N/A |
| 42%   74C    P2   185W / 250W |  10547MiB / 11172MiB |     84%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX 108...  Off  | 0000:82:00.0     Off |                  N/A |
| 31%   58C    P2   168W / 250W |  10547MiB / 11172MiB |     85%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      1563    G   /usr/lib/xorg/Xorg                             184MiB |
|    0     20836    G   compiz                                         171MiB |
|    0     22150    C   python3                                       5243MiB |
|    0     23382    C   python3                                       5301MiB |
|    1     22331    C   python3                                       5243MiB |
|    1     23440    C   python3                                       5301MiB |
|    2     22511    C   python3                                       5243MiB |
|    2     23753    C   python3                                       5301MiB |
|    3     22688    C   python3                                       5243MiB |
|    3     23811    C   python3                                       5301MiB |
+-----------------------------------------------------------------------------+
```

## Results

Make folders for storing result by:
```sh
./mkfolder.sh
```
### MNIST

#### AE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/mnist_ae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/mnist_ae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>


#### cluster AE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/mnist_clae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/mnist_clae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>

#### VAE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/mnist_vae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/mnist_vae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>


#### cluster VAE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/mnist_clvae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/mnist_clvae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>


### Sita
#### VAE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/sita_vae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/sita_vae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>

#### cluster VAE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/sita_clvae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/sita_clvae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>

### ShapeNet

#### AE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/shapenet_ae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/shapenet_ae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>

#### cluster AE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/shapenet_clae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/shapenet_clae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>

#### VAE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/shapenet_vae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/shapenet_vae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>

#### cluster VAE

<p>Samples for reconstruction and the reconstructed results:</p>
<table><tr>
<td><img src="readme_images/shapenet_clvae_test.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
<td><img src="readme_images/shapenet_clvae_recon.png" alt="HTML tutorial" style="width:256px;height:256px;border:0;"></td>
</tr></table>
