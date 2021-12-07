import argparse
import pickle
import os
from utils import mkdir

RESULT_DIR = 'D:/code/Image_WGAN_GP/results'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument('--exp_name', '-name', type=str,
                        help='output folder name; will be automatically generated if not specified')
    parser.add_argument('--pretrain_iterations', '-piters', type=int, default=2000, help='iterations for pre-training')
    parser.add_argument('--pretrain', action='store_true', default=False, help='if performing pre-training')
    parser.add_argument('--dataset', '-data', type=str, default='mnist', choices=['mnist', 'fashionmnist', 'cifar10'],
                        help=' dataset name')
    opt = parser.parse_args()
    return opt


def save_config(args):
    '''
    store the config and set up result dir
    :param args:
    :return:
    '''
    ### set up experiment name
    if args.exp_name is None:
        exp_name = '{}_epoch{}_batch_size{}_latent_dim{}_image_size{}_clip_value'.format(
            args.n_epochs,
            args.batch_size,
            args.latent_dim,
            args.img_size,
            args.clip_value
         )
        args.exp_name = exp_name

    if args.pretrain:
        save_dir = os.path.join(RESULT_DIR, args.dataset, 'pretrain', args.exp_name)
    else:
        save_dir = os.path.join(RESULT_DIR, args.dataset, 'main', args.exp_name)
    args.save_dir = save_dir

    ### save config
    mkdir(save_dir)
    config = vars(args)
    pickle.dump(config, open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in config.items():
            kv_str = k + ':' + str(v) + '\n'
            print(kv_str)
            f.writelines(kv_str)


def load_config(args):
    '''
    load the config
    :param args:
    :return:
    '''
    assert args.exp_name is not None, "Please specify the experiment name"
    if args.pretrain:
        save_dir = os.path.join(RESULT_DIR, args.dataset, 'pretrain', args.exp_name)
    else:
        save_dir = os.path.join(RESULT_DIR, args.dataset, 'main', args.exp_name)
    assert os.path.exists(save_dir)

    ### load config
    config = pickle.load(open(os.path.join(save_dir, 'params.pkl'), 'rb'))
    return config
