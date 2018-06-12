from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace, model_helper, net_drawer, memonger, brew
from caffe2.python import data_parallel_model as dpm
from caffe2.python.models import resnet
from caffe2.proto import caffe2_pb2
from matplotlib import pyplot
import numpy as np
import time
import os
import argparse
import imageio

base_learning_rate = 0.1
weight_decay = 1e-4
gpus = [0]



def accuracy(model):
    accuracy = []
    prefix = model.net.Proto().name
    for device in model._devices:
        accuracy.append(
            np.asscalar(workspace.FetchBlob("gpu_{}/{}_accuracy".format(device, prefix))))
    return np.average(accuracy)

def train_resnet50(args):
    # Model building functions
    def create_resnet50_model_ops(model, loss_scale=1.0):
        # Creates a residual network
        [softmax, loss] = resnet.create_resnet50(
            model,
            "data",
            num_input_channels=3,
            num_labels=1000,
            label="label",
            )
        prefix = model.net.Proto().name
        loss = model.net.Scale(loss, prefix + "_loss", scale=loss_scale)
        brew.accuracy(model, [softmax, "label"], prefix + "_accuracy")
        return [loss]

    # Create ModelHelper object
    train_arg_scope = {
        'order': 'NCHW',
        'use_gpu_engine': True,
        }
    train_model = model_helper.ModelHelper(
        name="resnet50", arg_scope=train_arg_scope
        )
    reader = train_model.CreateDB(
        "train_reader",
        db=args.train_data,
        db_type="lmdb")

    def add_image_input_ops(model):
        # utilize the ImageInput operator to prep the images
        data, label = brew.image_input(
            model,
            reader,
            ["data", "label"],
            batch_size=args.batch_size,
            # mean: to remove color values that are common
            mean=128.,
            # std is going to be modified randomly to influence the mean subtraction
            std=128.,
            # scale to rescale each image to a common size
            scale=256,
            # crop to the square each image to exact dimensions
            crop=224,
            # not running in test mode
            is_test=False,
            # mirroring of the images will occur randomly
            mirror=1,
            use_caffe_datum=False,
        )
        # prevent back-propagation: optional performance improvement; may not be observable at small scale
        data = model.net.StopGradient(data, data)

    def add_parameter_update_ops(model):
        brew.add_weight_decay(model, weight_decay)
        iter = brew.iter(model, "iter")
        lr = model.net.LearningRate(
            [iter],
            "lr",
            base_lr=base_learning_rate,
            policy="step",
            stepsize=int(10 * args.epochs_size / args.batch_size),
            gamma=0.1,
        )
        for param in model.GetParams():
            param_grad = model.param_to_grad[param]
            param_momentum = model.param_init_net.ConstantFill(
                [param], param + '_momentum', value=0.0
            )

            # Update param_grad and param_momentum in place
            model.net.MomentumSGDUpdate(
                [param_grad, param_momentum, lr, param],
                [param_grad, param_momentum, param],
                # almost 100% but with room to grow
                momentum=0.9,
                # netsterov is a defenseman for the Montreal Canadiens, but
                # Nesterov Momentum works slightly better than standard momentum
                nesterov=1,
            )

        
    dpm.Parallelize_GPU(
        train_model,
        input_builder_fun=add_image_input_ops,
        forward_pass_builder_fun=create_resnet50_model_ops,
        param_update_builder_fun=add_parameter_update_ops,
        devices=gpus,
        optimize_gradient_memory=True,
        cpu_device=True,
        )
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)
    #print(train_model.net.Proto())

    for epoch in range(args.num_epochs):
        num_iters = int(args.epochs_size / args.batch_size)
        for iter in range(num_iters):
            t1 = time.time()
            workspace.RunNet(train_model.net.Proto().name)
            t2 = time.time()
            dt = t2 - t1

            print((
                "Finished iteration {:>" + str(len(str(num_iters))) + "}/{}" +
                " (epoch {:>" + str(len(str(args.num_epochs))) + "}/{})" + 
                " ({:.2f} images/sec)").
                format(iter+1, num_iters, epoch+1, args.num_epochs, args.batch_size/dt))

            # Get the average accuracy for the training model
            train_accuracy = accuracy(train_model)
            print("Train accuracy: {:.3f}".format(train_accuracy))   

def main():
    parser = argparse.ArgumentParser(description="Caffe2: Resnet-50 training")
    parser.add_argument("--train_data", type=str, default=None, required=True, 
        help="Path to training data")
    parser.add_argument("--val_data", type=str, default=None, 
        help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=32,
        help="Batch size, total over all GPUs")
    parser.add_argument("--num_epochs", type=int, default=10,
        help="Num epochs.")
    parser.add_argument("--epochs_size", type=int, default=10000,
        help="Epoch size.")

    args = parser.parse_args()

    train_resnet50(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()

