import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
from utils import *
from models import *

ROOT_PATH = 'D:/code/Image_WGAN_GP'  # for windows


# ROOT_PATH = '../..' # for linux

def main(args):
    os.makedirs("images", exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    # Loss weight for gradient penalty
    lambda_gp = 10

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader
    if opt.dataset == 'mnist':
        os.makedirs(ROOT_PATH + "/data/mnist", exist_ok=True)

        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                ROOT_PATH + "/data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    elif opt.dataset == 'fashionmnist':
        os.makedirs(ROOT_PATH + "/data/fashionmnist", exist_ok=True)

        dataloader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                ROOT_PATH + "/data/fashionmnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    elif opt.dataset == 'cifar10':
        os.makedirs(ROOT_PATH + "/data/cifar10", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                ROOT_PATH + "data/cifar10",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                if batches_done % opt.sample_interval == 0:
                    save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += opt.n_critic


if __name__ == '__main__':
    opt = parse_arguments()
    save_config(opt)
    main(opt)
