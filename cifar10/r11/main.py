import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from model import *
from dataprocess import *
from tta_utils import *
import os
import time


parser = argparse.ArgumentParser(description='CIFAR10 SNN TTA')
parser.add_argument('--data', default='/path/to/cifar10',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=512, type=int,
                    metavar='N')
parser.add_argument('-T', '--timesteps', default=30, type=int,
                    help='Simulation timesteps')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--domain', default='cifar10', type=str)
parser.add_argument('--phase', default=None, type=str)
parser.add_argument('--method', default='space', type=str)
# 0.005 for memo, 0.1 for ls
parser.add_argument('--lr', default=0.1, type=float)


def test_single_image(net, img_test, label, T, accuracy, test_num):
    net.eval()
    with torch.no_grad():
        label = label.cuda()
        img_test = img_test.cuda()
        output, _ = net_forward(net, img_test, T, phase='test')

        accuracy += (output.argmax(dim=1) ==
                     label).float().sum().item()
        test_num += label.numel()
        net.reset_()

    return accuracy, test_num


def main():
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    T = args.timesteps

    cudnn.benchmark = True

    # Load data
    test_data_loader, adapt_set = load_data(args)

    # Test / Adapt
    if args.phase == 'test':
        # Prepare model
        net = ResNet11().cuda()
        checkpoint = torch.load(
            './model_bestT1_cifar10_r11.pth.tar', weights_only=False)
        net.load_state_dict(checkpoint['state_dict'])
        # print(net)

        with torch.no_grad():
            print(f'Test on {args.domain}:')
            net.eval()
            accuracy = 0
            test_num = 0
            for img, label in tqdm(test_data_loader):
                label = label.cuda()
                img = img.cuda()

                for t in range(T - 1):
                    # Poisson encoding
                    rand_num = torch.rand_like(img).cuda()
                    poisson_input = (torch.abs(img) > rand_num).float()
                    poisson_input = torch.mul(poisson_input, torch.sign(img))

                    net(poisson_input)

                output, _ = net(poisson_input)

                accuracy += (output.argmax(dim=1) ==
                             label).float().sum().item()
                test_num += label.numel()
                net.reset_()

            accuracy /= test_num
            print(f'Test Acc on {args.domain}: {accuracy}')

    elif args.phase == 'adapt':
        results_dir = './results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        timestamp = time.strftime("%Y%m%d-%H%M")
        results_file = os.path.join(
            results_dir, f'{timestamp}_{args.domain}_adaptation.txt')

        checkpoint = torch.load(
            './model_bestT1_cifar10_r11.pth.tar', weights_only=False)
        accuracy = 0
        test_num = 0

        print(f'Adapt on {args.domain}:')
        with open(results_file, 'a') as f:
            f.write(f'Adapt on {args.domain}:\n')

        net = ResNet11().cuda()

        for j in range(len(adapt_set)):
            img_np, label = adapt_set[j]
            img = Image.fromarray(img_np)
            img_test = test_transform(img).unsqueeze(0)
            label = torch.tensor(label)

            if j % 100 == 0:
                start_time = time.time()

            optimizer = torch.optim.SGD(
                net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

            net.load_state_dict(checkpoint['state_dict'])

            # Adaptation
            net.eval()
            for _ in range(1):
                images = [augmix(img) for _ in range(32)]
                images = torch.stack(images).cuda()

                output, feature_map = net_forward(
                    net, images, T, phase='train')

                loss = einsum(feature_map)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                net.reset_()

            accuracy, test_num = test_single_image(
                net, img_test, label, T, accuracy, test_num)

            del optimizer
            torch.cuda.empty_cache()

            if (j + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()

                # Write the results to the file
                with open(results_file, 'a') as f:
                    f.write('{} test points on {} domain: {:.2f}%, taking {:.2f} seconds\n'.format(
                        j + 1, args.domain, accuracy / test_num * 100, elapsed_time))
                print('{} test points on {} domain: {:.2f}%, taking {:.2f} seconds'.format(
                    j + 1, args.domain, accuracy / test_num * 100, elapsed_time))

        accuracy /= test_num
        with open(results_file, 'a') as f:
            f.write(f'TTA Acc on {args.domain}: {accuracy}\n')
        print(f'TTA Acc on {args.domain}: {accuracy}')


if __name__ == '__main__':
    main()
