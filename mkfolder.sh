## Making directories for storing results
mkdir result_mnist_clae
mkdir result_mnist_ae
mkdir result_mnist_clvae
mkdir result_mnist_vae
mkdir result_sita_clae
mkdir result_sita_ae
mkdir result_sita_clvae
mkdir result_sita_vae
mkdir result_shapenet_clae
mkdir result_shapenet_ae
mkdir result_shapenet_clvae
mkdir result_shapenet_vae

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

## Move montage figures into a target folder
# mnist samples
cp ../../buildboat/cluster-vae/result_mnist_ae/test_xs.png ./readme_images/mnist_ae_test.png
cp ../../buildboat/cluster-vae/result_mnist_vae/test_xs.png ./readme_images/mnist_vae_test.png
cp ../../buildboat/cluster-vae/result_mnist_clvae/test_xs.png ./readme_images/mnist_clvae_test.png
cp ../../buildboat/cluster-vae/result_mnist_clae/test_xs.png ./readme_images/mnist_clae_test.png
# shapenet samples
cp ../../buildboat/cluster-vae/result_shapenet_clae/test_xs.png ./readme_images/shapenet_clae_test.png
cp ../../buildboat/cluster-vae/result_shapenet_clvae/test_xs.png ./readme_images/shapenet_clvae_test.png
cp ../../buildboat/cluster-vae/result_shapenet_vae/test_xs.png ./readme_images/shapenet_vae_test.png
cp ../../buildboat/cluster-vae/result_shapenet_ae/test_xs.png ./readme_images/shapenet_ae_test.png
# mnist result
cp ../../buildboat/cluster-vae/result_mnist_ae/reconstruction_latest.png ./readme_images/mnist_ae_recon.png
cp ../../buildboat/cluster-vae/result_mnist_vae/reconstruction_latest.png ./readme_images/mnist_vae_recon.png
cp ../../buildboat/cluster-vae/result_mnist_clvae/reconstruction_latest.png ./readme_images/mnist_clvae_recon.png
cp ../../buildboat/cluster-vae/result_mnist_clae/reconstruction_latest.png ./readme_images/mnist_clae_recon.png
# shapenet result
cp ../../buildboat/cluster-vae/result_shapenet_clae/reconstruction_latest.png ./readme_images/shapenet_clae_recon.png
cp ../../buildboat/cluster-vae/result_shapenet_clvae/reconstruction_latest.png ./readme_images/shapenet_clvae_recon.png
cp ../../buildboat/cluster-vae/result_shapenet_vae/reconstruction_latest.png ./readme_images/shapenet_vae_recon.png
cp ../../buildboat/cluster-vae/result_shapenet_ae/reconstruction_latest.png ./readme_images/shapenet_ae_recon.png
