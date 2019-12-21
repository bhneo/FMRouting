import os
import argparse


def run(args):
    os.system(
        'python ./main_multi.py --idx={} --model={} --pool={} --atoms={} --recons_conv=True --recons_share=True --batch={} --gpu={} --steps=80000'.format(
            str(args.idx), args.model, 'FM', str(args.atoms), args.batch_size, str(args.gpu)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--idx', default=1, help='')
    parser.add_argument('--model', default='ex4_3_2', help='which model to run')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--pool', default='routing', help='routing')
    parser.add_argument('--atoms', default=16, type=int, help='atoms per capsule')
    parser.add_argument('--batch_size', default=128, type=int, help='')
    arguments = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    run(arguments)
