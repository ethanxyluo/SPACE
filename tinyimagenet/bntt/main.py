#############################################
#   @author: Youngeun Kim and Priya Panda   #
#############################################
# --------------------------------------------------
# Imports
# --------------------------------------------------
from utills import *
import torch.backends.cudnn as cudnn
import numpy as np
import os.path
import argparse
import torch.optim as optim
import torchvision
from torchvision import transforms
from model import *
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
    description='SNN trained with BNTT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed',                  default=0,
                    type=int,   help='Random seed')
parser.add_argument('--num_steps',             default=30,
                    type=int, help='Number of time-step')
parser.add_argument('--batch_size',            default=64,
                    type=int,   help='Batch size')
parser.add_argument('--leak_mem',              default=0.99,
                    type=float, help='Leak_mem')
parser.add_argument('--num_workers',           default=4,
                    type=int, help='number of workers')
parser.add_argument('--domain',                default='tinyimagenet',
                    type=str, help='domain')
parser.add_argument('--gpu',                   default='3',
                    type=str, help='gpu')
parser.add_argument('--phase',                 default='test',
                    type=str, help='phase [test, adapt]')
parser.add_argument('--method',                  default='space',
                    type=str, help='method')
parser.add_argument('--lr',                    default=0.5,
                    type=float, help='learning rate')

global args
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# --------------------------------------------------
# Initialize tensorboard setting
# --------------------------------------------------
ckpt_path = '/path/to/tinyimagenet_vgg11.pth.tar'


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
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

num_cls = 200
img_size = 64

if args.domain == 'tinyimagenet':
    val_set = torchvision.datasets.ImageFolder(
        root='/path/to/tiny-imagenet-200/val', transform=transform_test)
else:
    domaindir = os.path.join(
        '/path/to/Tiny-ImageNet-C', args.domain, '5')
    adapt_set = torchvision.datasets.ImageFolder(
        root=domaindir, transform=None)
    val_set = torchvision.datasets.ImageFolder(
        root=domaindir, transform=transform_test)

testloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_test,
                                         shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)


if args.phase == 'test':
    print('********** SNN testing **********')
    model = SNN_VGG11_BNTT(
        num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls).cuda()
    modelsave = torch.load(ckpt_path)
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
    modelsave = torch.load(ckpt_path)
    acc_top1, acc_top5 = [], []

    # Create the results directory if it doesn't exist
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Create the file path with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M")
    results_file = os.path.join(
        results_dir, f'{timestamp}_{args.domain}_adaptation.txt')

    # Adaptation
    for j in range(len(adapt_set)):
        if j % 100 == 0:
            start_time = time.time()

        image, label = adapt_set[j]
        image_test = transform_test(image).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)

        model = SNN_VGG11_BNTT(num_steps=num_steps, leak_mem=leak_mem,
                               img_size=img_size,  num_cls=num_cls).cuda()
        model.load_state_dict(modelsave['state_dict'])

        optimizer_adapt = optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=1e-4)

        # # Adaptation
        model = adapt(model, optimizer_adapt, image, niter=1,
                      batch_size=32, method=args.method)

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
