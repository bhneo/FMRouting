from easydict import EasyDict
from common import utils

import argparse

params = EasyDict()

params.logdir = 'log'  # dir of logs
params.gpu = 1

params.dataset = EasyDict()
params.dataset.name = 'cifar10'
params.dataset.flip = True
params.dataset.crop = True

params.training = EasyDict()
params.training.batch_size = 128
params.training.epochs = 163
params.training.steps = 9999999  # The number of training steps
params.training.lr_steps = [30000, 40000]
params.training.verbose = True
params.training.log_steps = 1000
params.training.idx = 1
params.training.momentum = 0.9
params.training.save_frequency = 0
params.training.log = True
params.training.whiten = 'zca'

params.routing = EasyDict()
params.routing.iter_num = 3  # number of iterations in routing algorithm
params.routing.temper = 1  # the lambda in softmax
params.routing.input_norm = False
params.routing.accumulate = True
params.routing.use_normed = False
params.routing.inner = True

params.caps = EasyDict()
params.caps.atoms = 8  # number of atoms in a capsule
params.caps.pre_atoms = 8
params.caps.regularization_scale = 0.0005  # regularization coefficient for reconstruction loss, default to 0.0005*784=0.392

params.model = EasyDict()
params.model.name = 'resnet.resnet_agreement_multi'  # which model to use
params.model.pool = 'FM'  # avg, max, routing, FM
params.model.layer_num = 20
params.model.in_norm = True
params.model.in_norm_fn = 'norm'
params.model.fm_norm = 'constant4'
params.model.out_norm = True
params.model.norm_scale = True
params.model.bn_relu = True
params.model.resnet = 'v2'

params.recons = EasyDict()
params.recons.balance_factor = 0.0005
params.recons.threshold = 0.8
params.recons.conv = False
params.recons.share = True


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--gpu', default='0', help='which gpu to use')
    parser.add_argument('--train', default=True, help='train of evaluate')
    parser.add_argument('--t_log', default=True, help='tensorboard log')
    parser.add_argument('--dataset', default=params.dataset.name, help='dataset config')
    parser.add_argument('--flip', default=params.dataset.flip, help='dataset config')
    parser.add_argument('--crop', default=params.dataset.crop, help='dataset config')
    parser.add_argument('--model', default=params.model.name, help='network config')
    parser.add_argument('--pool', default=params.model.pool, help='the type of pooling layer')
    parser.add_argument('--idx', default=1, help='the index of trial')
    parser.add_argument('--epochs', default=params.training.epochs, help='the total training epochs')
    parser.add_argument('--batch', default=params.training.batch_size, help='the training batch_size')
    parser.add_argument('--steps', default=params.training.steps, help='the total training steps')
    parser.add_argument('--whiten', default=params.training.whiten, help='method to whiten')
    parser.add_argument('--log', default=params.logdir, help='directory to save log')
    parser.add_argument('--log_steps', default=params.training.log_steps, help='frequency to log by steps')
    parser.add_argument('--layer_num', default=params.model.layer_num, help='the number of layers')
    parser.add_argument('--in_norm', default=params.model.in_norm, help='do vector norm before pooling')
    parser.add_argument('--in_norm_fn', default=params.model.in_norm_fn, help='activation before pooling')
    parser.add_argument('--fm_norm', default=params.model.fm_norm, help='norm for fm')
    parser.add_argument('--out_norm', default=params.model.in_norm, help='do vector norm after pooling')
    parser.add_argument('--norm_scale', default=params.model.norm_scale, help='do vector norm before pooling')
    parser.add_argument('--bn_relu', default=params.model.bn_relu, help='do bn_relu or only bn')
    parser.add_argument('--resnet', default=params.model.resnet, help='resnet version')
    parser.add_argument('--temper', default=params.routing.temper, help='the lambda in softmax')
    parser.add_argument('--iter_num', default=params.routing.iter_num, help='the iter num of routing')
    parser.add_argument('--atoms', default=params.caps.atoms, help='capsule atoms')
    parser.add_argument('--pre_atoms', default=params.caps.pre_atoms, help='pri-caps atoms')
    parser.add_argument('--balance_factor', default=params.recons.balance_factor, help='Loss factor of reconstruction')
    parser.add_argument('--activate_threshold', default=params.recons.threshold, help='The threshold of activation')
    parser.add_argument('--recons_conv', default=params.recons.conv, help='Use conv layers for reconstruction')
    parser.add_argument('--recons_share', default=params.recons.share, help='Share capsules to do reconstruction')
    arguments = parser.parse_args()
    build_params = build_config(arguments, params)
    return arguments, build_params


def build_config(args, build_params):
    build_params.gpu = args.gpu
    build_params.logdir = args.log
    build_params.dataset.name = args.dataset
    build_params.dataset.flip = utils.str2bool(args.flip)
    build_params.dataset.crop = utils.str2bool(args.crop)
    build_params.training.log_steps = int(args.log_steps)
    build_params.training.idx = args.idx
    build_params.training.epochs = int(args.epochs)
    build_params.training.batch_size = int(args.batch)
    build_params.training.steps = int(args.steps)
    build_params.training.log = utils.str2bool(args.t_log)
    build_params.training.whiten = args.whiten
    build_params.model.name = args.model
    build_params.model.pool = args.pool
    build_params.model.layer_num = int(args.layer_num)
    build_params.model.in_norm = utils.str2bool(args.in_norm)
    build_params.model.in_norm_fn = args.in_norm_fn
    build_params.model.fm_norm = args.fm_norm
    build_params.model.out_norm = utils.str2bool(args.out_norm)
    build_params.model.norm_scale = utils.str2bool(args.norm_scale)
    build_params.model.bn_relu = utils.str2bool(args.bn_relu)
    build_params.model.resnet = args.resnet
    build_params.routing.temper = float(args.temper)
    build_params.routing.iter_num = int(args.iter_num)
    build_params.caps.atoms = int(args.atoms)
    build_params.caps.pre_atoms = int(args.pre_atoms)
    build_params.recons.balance_factor = float(args.balance_factor)
    build_params.recons.threshold = float(args.activate_threshold)
    build_params.recons.conv = utils.str2bool(args.recons_conv)
    build_params.recons.share = utils.str2bool(args.recons_share)
    return build_params





