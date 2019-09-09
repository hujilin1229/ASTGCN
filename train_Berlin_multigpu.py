# -*- coding:utf-8 -*-

import os
import shutil
from time import time
from datetime import datetime
import configparser
import argparse

import numpy as np

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxboard import SummaryWriter

from lib.utils import compute_val_loss_multigpu, evaluate_multigpu, predict_multigpu
from lib.data_preparation import read_and_generate_dataset, read_and_generate_dataset_from_files
from model.model_config import get_backbones, get_backbones_traffic4cast

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    help="configuration file path", required=True)
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
args = parser.parse_args()


os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = '0'
# mxboard log dir
if os.path.exists('logs'):
    shutil.rmtree('logs')
    print('Remove log dir')

# read configuration
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
node_pos_filename = data_config['node_pos_filename']
data_dir = data_config['data_dir']

# num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])

model_name = training_config['model_name']
# ctx = training_config['ctx']
optimizer = training_config['optimizer']
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
merge = bool(int(training_config['merge']))

# n_gpu = mx.context.num_gpus()
gpus = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
# select devices
# if ctx.startswith('cpu'):
#     ctx = mx.cpu()
# elif ctx.startswith('gpu'):
#     ctx = mx.gpu(int(ctx[ctx.index('-') + 1:]))
#     print("using GPU: ", ctx)
ctx = [mx.gpu(i) for i in gpus] if len(gpus) >= 1 else \
          [mx.cpu()]


# import model
print('Model is %s' % (model_name))
if model_name == 'MSTGCN':
    from model.mstgcn import MSTGCN as model
elif model_name == 'ASTGCN':
    from model.astgcn_smaller import ASTGCN as model
else:
    raise SystemExit('Wrong type of model!')

# make model params dir
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if 'params_dir' in training_config and training_config['params_dir'] != "None":
    params_path = os.path.join(training_config['params_dir'], model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)

# check parameters file
if os.path.exists(params_path) and not args.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s' % (params_path))


class MyInit(mx.init.Initializer):
    xavier = mx.init.Xavier()
    uniform = mx.init.Uniform()

    def _init_weight(self, name, data):
        if len(data.shape) < 2:
            self.uniform._init_weight(name, data)
            print('Init', name, data.shape, 'with Uniform')
        else:
            self.xavier._init_weight(name, data)
            print('Init', name, data.shape, 'with Xavier')

if __name__ == "__main__":
    # read all data from graph signal matrix file
    print("Reading data...")
    node_pos = np.load(node_pos_filename)
    num_of_vertices = node_pos.shape[0]
    all_data = read_and_generate_dataset_from_files(data_dir,
                                                    node_pos,
                                                    num_of_weeks,
                                                    num_of_days,
                                                    num_of_hours,
                                                    num_for_predict,
                                                    points_per_hour,
                                                    merge)

    # test set ground truth,
    # the original target shape is (#batch, #vertices, #pred)
    # the transposed target shape is (#batch, #pred, #vertices)
    true_value = (all_data['test']['target'].transpose((0, 2, 3, 1))
                  .reshape(all_data['test']['target'].shape[0], -1))

    # reshape the target data for all sets
    all_data['train']['target'] = all_data['train']['target'].reshape(
        all_data['train']['target'].shape[0], all_data['train']['target'].shape[1], -1)
    all_data['val']['target'] = all_data['val']['target'].reshape(
        all_data['val']['target'].shape[0], all_data['val']['target'].shape[1], -1)

    # training set data loader
    train_loader = gluon.data.DataLoader(
                        gluon.data.ArrayDataset(
                            nd.array(all_data['train']['week']),
                            nd.array(all_data['train']['day']),
                            nd.array(all_data['train']['recent']),
                            nd.array(all_data['train']['target'])
                        ),
                        batch_size=batch_size,
                        shuffle=True
    )

    # validation set data loader
    val_loader = gluon.data.DataLoader(
                    gluon.data.ArrayDataset(
                        nd.array(all_data['val']['week']),
                        nd.array(all_data['val']['day']),
                        nd.array(all_data['val']['recent']),
                        nd.array(all_data['val']['target'])
                    ),
                    batch_size=batch_size,
                    shuffle=False
    )

    # testing set data loader
    test_loader = gluon.data.DataLoader(
                    gluon.data.ArrayDataset(
                        nd.array(all_data['test']['week']),
                        nd.array(all_data['test']['day']),
                        nd.array(all_data['test']['recent']),
                        nd.array(all_data['test']['target'])
                    ),
                    batch_size=batch_size,
                    shuffle=False
    )

    # save Z-score mean and std
    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean']
        stats_data[type_ + '_std'] = stats['std']
    np.savez_compressed(
        os.path.join(params_path, 'stats_data'),
        **stats_data
    )

    # loss function MSE
    loss_function = gluon.loss.L2Loss()
    metric = mx.metric.MSE()
    # get model's structure
    ctx1 = ctx[0]
    all_backbones, cheb_polynomials = get_backbones_traffic4cast(args.config, adj_filename)


    """Model initialization."""
    # kwargs = {'ctx': ctx}
    net = model(num_for_predict, all_backbones)
    net.initialize(ctx=ctx)
    # net.collect_params().initialize(ctx=ctx)
    # print(net, flush=True)
    # print(net.collect_params(), flush=True)

    for index, (val_w, val_d, val_r, _) in enumerate(val_loader):
        val_w = gluon.utils.split_and_load(val_w, ctx_list=ctx, even_split=False)
        val_d = gluon.utils.split_and_load(val_d, ctx_list=ctx, even_split=False)
        val_r = gluon.utils.split_and_load(val_r, ctx_list=ctx, even_split=False)
        outputs = [net(x_list=[w, d, r], cheb_polynomials=cheb_polynomials) for w, d, r in zip(val_w, val_d, val_r)]
        print("test printing: ", outputs[0])

    net.initialize(ctx=ctx, init=MyInit(), force_reinit=True)

    # initialize a trainer to train model
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': learning_rate})

    # initialize a SummaryWriter to write information into logs dir
    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    # compute validation loss before training
    compute_val_loss_multigpu(net, val_loader, loss_function, sw,
                              epoch=0, ctx=ctx, cheb_polynomials=cheb_polynomials)

    # compute testing set MAE, RMSE, MAPE before training
    evaluate_multigpu(net, test_loader, true_value, num_of_vertices, sw,
                      epoch=0, ctx=ctx, cheb_polynomials=cheb_polynomials)

    # train model
    global_step = 1
    for epoch in range(1, epochs + 1):

        for train_w, train_d, train_r, train_t in train_loader:
            actual_batch_size = train_w.shape[0]
            # running on multi-gpus
            train_w = gluon.utils.split_and_load(train_w, ctx_list=ctx, even_split=False)
            train_d = gluon.utils.split_and_load(train_d, ctx_list=ctx, even_split=False)
            train_r = gluon.utils.split_and_load(train_r, ctx_list=ctx, even_split=False)
            train_t = gluon.utils.split_and_load(train_t, ctx_list=ctx, even_split=False)
            start_time = time()
            with autograd.record():
                outputs = [net(x_list=[w, d, r], cheb_polynomials=cheb_polynomials) for w, d, r in zip(train_w, train_d, train_r)]
                losses = [loss_function(o, l) for o, l in zip(outputs, train_t)]
            for loss in losses:
                loss.backward()

            # update metric for each output
            for label, o in zip(train_t, outputs):
                metric.update(label, o)

            # Update the parameters by stepping the trainer; the batch size
            # is required to normalize the gradients by `1 / batch_size`.
            trainer.step(batch_size=actual_batch_size, ignore_stale_grad=True)

            name, acc = metric.get()

            sw.add_scalar(tag='training_loss',
                          value=acc,
                          global_step=global_step)

            print('global step: %s, training loss: %.2f, time: %.2fs'
                  % (global_step, acc, time() - start_time))
            global_step += 1

        # logging the gradients of parameters for checking convergence
        for name, param in net.collect_params().items():
            try:
                sw.add_histogram(tag=name + "_grad",
                                 values=param.grad(),
                                 global_step=global_step,
                                 bins=1000)
            except:
                print("can't plot histogram of {}_grad".format(name))

        # compute validation loss
        compute_val_loss_multigpu(net, val_loader, loss_function, sw,
                                  epoch, ctx=ctx, cheb_polynomials=cheb_polynomials)

        # evaluate the model on testing set
        evaluate_multigpu(net, test_loader, true_value, num_of_vertices, sw,
                          epoch, ctx=ctx, cheb_polynomials=cheb_polynomials)

        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s.params' % (model_name,
                                                               epoch))
        net.save_parameters(params_filename)
        print('save parameters to file: %s' % (params_filename))

    # close SummaryWriter
    sw.close()

    if 'prediction_filename' in training_config:
        prediction_path = training_config['prediction_filename']

        prediction = predict_multigpu(net, test_loader, ctx=ctx)

        np.savez_compressed(
            os.path.normpath(prediction_path),
            prediction=prediction,
            ground_truth=all_data['test']['target']
        )
