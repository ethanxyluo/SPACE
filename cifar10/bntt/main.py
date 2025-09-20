
from PIL import Image
from utills import *
import torch.backends.cudnn as cudnn
import numpy as np
import os.path
import argparse
import torch.optim as optim
import torchvision
from torchvision import transforms
from model import *
from dataprocess import CIFAR10C
from adaptation.test_adaptation import *
import warnings
import sys
import time
warnings.filterwarnings('ignore')

cudnn.benchmark = True
cudnn.deterministic = True


# --------------------------------------------------
# Parse input arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(
    description='SNN with BNTT TTA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed',                  default=0,
                    type=int,   help='Random seed')
parser.add_argument('--num_steps',             default=25,
                    type=int, help='Number of time-step')
parser.add_argument('--batch_size',            default=64,
                    type=int,   help='Batch size')
parser.add_argument('--batch_size_adapt',      default=32,
                    type=int,   help='Batch size for adaptation')
parser.add_argument('--leak_mem',              default=0.95,
                    type=float, help='Leak_mem')
parser.add_argument('--num_workers',           default=4,
                    type=int, help='number of workers')
parser.add_argument('--domain',                default='cifar10',
                    type=str, help='domain')
parser.add_argument('--gpu',                   default='2',
                    type=str, help='gpu')
parser.add_argument('--phase',                 default='test',
                    type=str, help='phase [test, adapt]')
parser.add_argument('--method',                default='space',
                    type=str, help='method')
parser.add_argument('--lr',                    default=0.5,
                    type=float, help='learning rate')
parser.add_argument('--l',                     default=1,
                    type=int, help='augment strength')

global args
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# --------------------------------------------------
# Initialize tensorboard setting
# --------------------------------------------------
log_dir = 'modelsave'
if os.path.isdir(log_dir) is not True:
    os.mkdir(log_dir)


user_foldername = 'cifar10vgg9_timestep25_lr0.3_epoch100_leak0.95'


# --------------------------------------------------
# Initialize seed
# --------------------------------------------------
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --------------------------------------------------
# SNN configuration parameters
# --------------------------------------------------
# Leaky-Integrate-and-Fire (LIF) neuron parameters
leak_mem = args.leak_mem

# SNN learning and evaluation parameters
batch_size_test = args.batch_size
num_steps = args.num_steps


# --------------------------------------------------
# Load  dataset
# --------------------------------------------------
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
        0.2023, 0.1994, 0.2010])])

num_cls = 10
img_size = 32
if args.domain == 'cifar10':
    test_set = torchvision.datasets.CIFAR10(root='/path/to/cifar10', train=False,
                                            download=True, transform=transform_test)
else:
    adapt_set = CIFAR10C(root='/path/to/cifar10/CIFAR-10-C', name=args.domain,
                         transform=None, level=5)
    test_set = CIFAR10C(root='/path/to/cifar10/CIFAR-10-C', name=args.domain,
                        transform=transform_test, level=5)

testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test,
                                         shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)


if args.phase == 'test':
    print('********** SNN testing **********')
    model = SNN_VGG9_BNTT(num_steps=num_steps, leak_mem=leak_mem,
                          img_size=img_size,  num_cls=num_cls).cuda()
    modelsave = torch.load(log_dir + '/' + user_foldername +
                           '_bestmodel.pth.tar')
    model.load_state_dict(modelsave['state_dict'])
    acc_top1, acc_top5 = [], []

    model.eval()
    with torch.no_grad():
        for j, data in enumerate(testloader, 0):

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            out, _ = model(images)

            prec1, prec5 = accuracy(out, labels, topk=(1, 5))
            acc_top1.append(float(prec1))
            acc_top5.append(float(prec5))

    test_accuracy = np.mean(acc_top1)
    print("Test accuracy on {} domain: {:.2f}%".format(
        args.domain, test_accuracy))

elif args.phase == 'adapt':
    print('********** SNN adaptation and testing **********')
    modelsave = torch.load(log_dir + '/' + user_foldername +
                           '_bestmodel.pth.tar')['state_dict']
    acc_top1, acc_top5 = [], []

    # Create the results directory if it doesn't exist
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Create the file path with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M")
    results_file = os.path.join(
        results_dir, f'{timestamp}_{args.domain}_{args.method}_{args.lr}.txt')

    # Adaptation
    for j in range(len(adapt_set)):
        if j % 100 == 0:
            start_time = time.time()

        image_np, label = adapt_set[j]
        image = Image.fromarray(image_np)
        image_test = transform_test(image).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)

        model = SNN_VGG9_BNTT(num_steps=num_steps, leak_mem=leak_mem,
                              img_size=img_size,  num_cls=num_cls).cuda()
        model.load_state_dict(modelsave)

        optimizer_adapt = optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=1e-4)

        # Adaptation
        model = adapt(model, optimizer_adapt, image, l=args.l,
                      niter=1, batch_size=args.batch_size_adapt)

        model.eval()
        with torch.no_grad():
            image_test = image_test.cuda()
            label = label.cuda()
            out, _ = model(image_test)
            prec1, prec5 = accuracy(out, label, topk=(1, 5))
            acc_top1.append(float(prec1))
            acc_top5.append(float(prec5))

        if (j + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()

            # Write the results to the file
            with open(results_file, 'a') as f:
                f.write('{} test points on {} domain: {:.2f}%, taking {:.2f} seconds\n'.format(
                    args.domain, j + 1, np.mean(acc_top1), elapsed_time))
            print('{} test points on {} domain: {:.2f}%, taking {:.2f} seconds'.format(
                args.domain, j + 1, np.mean(acc_top1), elapsed_time))

    test_accuracy = np.mean(acc_top1)
    with open(results_file, 'a') as f:
        f.write("Test accuracy on {} domain: {:.2f}%\n".format(
            args.domain, test_accuracy))
    print("Test accuracy on {} domain: {:.2f}%".format(
        args.domain, test_accuracy))

sys.exit(0)
