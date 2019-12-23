import os
import argparse


def run(args):
    idx=int(args.gpu)+1
    # os.system(
    #     'python models/small_norb.py --idx={} --pool={} --iter_num={} --dataset={} --atoms={} --batch={} --gpu={} --epochs=200'.format(
    #         idx, args.pool, str(args.iter), args.dataset, str(args.atoms), args.batch_size, str(args.gpu)))
    os.system(
        'python models/ex4_4.py --idx={} --pool={} --iter_num={} --dataset={} --atoms={} --batch={} --gpu={} --epochs=200'.format(
            idx, 'FM', str(args.iter), args.dataset, str(args.atoms), args.batch_size, str(args.gpu)))
    os.system(
        'python models/ex4_4.py --idx={} --pool={} --iter_num={} --dataset={} --atoms={} --batch={} --gpu={} --epochs=200'.format(
            idx, 'EM', str(1), args.dataset, str(args.atoms), args.batch_size, str(args.gpu)))
    os.system(
        'python models/ex4_4.py --idx={} --pool={} --iter_num={} --dataset={} --atoms={} --batch={} --gpu={} --epochs=200'.format(
            idx, 'EM', str(2), args.dataset, str(args.atoms), args.batch_size, str(args.gpu)))
    os.system(
        'python models/ex4_4.py --idx={} --pool={} --iter_num={} --dataset={} --atoms={} --batch={} --gpu={} --epochs=200'.format(
            idx, 'EM', str(3), args.dataset, str(args.atoms), args.batch_size, str(args.gpu)))
    os.system(
        'python models/ex4_4.py --idx={} --pool={} --iter_num={} --dataset={} --atoms={} --batch={} --gpu={} --epochs=200'.format(
            idx, 'dynamic', str(1), args.dataset, str(args.atoms), args.batch_size, str(args.gpu)))
    os.system(
        'python models/ex4_4.py --idx={} --pool={} --iter_num={} --dataset={} --atoms={} --batch={} --gpu={} --epochs=200'.format(
            idx, 'dynamic', str(2), args.dataset, str(args.atoms), args.batch_size, str(args.gpu)))
    os.system(
        'python models/ex4_4.py --idx={} --pool={} --iter_num={} --dataset={} --atoms={} --batch={} --gpu={} --epochs=200'.format(
            idx, 'dynamic', str(3), args.dataset, str(args.atoms), args.batch_size, str(args.gpu)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--model', default='res_cifar.agreement3_norb4', help='which model to run')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--iter', default=3, type=int, help='iter num')
    parser.add_argument('--pool', default='EM', help='routing')
    parser.add_argument('--atoms', default=16, type=int, help='atoms per capsule')
    parser.add_argument('--dataset', default='smallNORB', help='norm of fm-out')
    parser.add_argument('--batch_size', default=64, type=int, help='')
    arguments = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    run(arguments)
