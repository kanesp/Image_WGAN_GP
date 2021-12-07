# Image_WGAN_GP
This repository implements WGAN_GP. 
# Image_WGAN_GP

This repository uses wgan to generate mnist and fashionmnist pictures. Firstly, you can download the datasets from  `main.py` .

## requirements

Before you run the code, you should install following packages for your environment.

You can see it in the `requirements.txt`.

### install

`pip install -r requirements.txt`

```shell
torch>=0.4.0
torchvision
matplotlib
numpy
scipy
pillow
urllib3
scikit-image
```

## Prepare the dataset

Before  you run the code, you should prepare the dataset.  You must replace  the `ROOT_PATH` in `main.py` with your own path.

```shell
ROOT_PATH = '../..' # for linux
ROOT_PATH = 'D:/code/Image_WGAN_GP'  # for windows 
```

We provide the mnist , fashionmnist and cifar10 datasets. But you can download others , when you run the code.  For example , download the cifar100, just add the following code in `main.py` and  **you should modify the models(We will finish it later)**. 

```
opt.dataset == 'cifar100':
    os.makedirs(ROOT_PATH + "/data/cifar100", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            ROOT_PATH + "data/cifar100",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
```

The data will be saved in `data` directory.

## Training 

Using mnist dataset.

```train
python main.py -data 'mnist' -n_epochs 300
```

Using fashionmnist dataset.

` python main.py -data 'fashionmnist' -n_epochs 300 `

The generated images will be saved in `images` directory.

###  Training parameters 

You can see details in `config.py`

```shell
"--n_epochs", "number of epochs of training"

"--batch_size", "size of the batches"

"--lr","adam: learning rate"

"--b1","adam: decay of first order momentum of gradient"

"--b2", "adam: decay of first order momentum of gradient"

"--n_cpu", "number of cpu threads to use during batch generation"

"--latent_dim", "dimensionality of the latent space"

"--img_size", "size of each image dimension"

"--channels","number of image channels"

"--n_critic", "number of training steps for discriminator per iter"

"--clip_value","lower and upper clip value for disc. weights"

"--sample_interval", "interval betwen image samples"

'--exp_name', 'output folder name; will be automatically generated if not specified'

'--pretrain_iterations', 'iterations for pre-training'

'--pretrain', 'if performing pre-training'

'--dataset', '-data', choices=['mnist', 'fashionmnist', 'cifar10']
```



### Save params

The parameters will be  save in results.  And you can change the saving directory name in `config.py`

## Wasserstein GAN GP

*Improved Training of Wasserstein GANs*

#### Authors

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville

#### Abstract

Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only low-quality samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

[[Paper\]](https://arxiv.org/abs/1704.00028) 

![wgan_gp](https://github.com/kanesp/Image_WGAN_GP/blob/main/wgan_gp.gif?raw=true)

