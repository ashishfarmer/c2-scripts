from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from matplotlib import pyplot as plt
import time
import math
import numpy as np
import os
import lmdb
import shutil
from imageio import imread
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.proto import caffe2_pb2
from caffe2.python import data_parallel_model as dpm
from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    visualize,
    workspace,
    scope,
)

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=2', '--caffe2_gpu_memory_tracking=0'])
print("Necessities imported!")

# Parameters for training - can take these as command line args
#train_data = '/home/ashish/data/VAL_notraw'
train_data = '/home/ashish/caffe2_notebooks/tutorial_data/resnet_trainer/imagenet_cars_boats_train'
num_epochs = 10
epoch_size = 1000
gpus = [0]
image_size = 224
batch_size = 32

def AddAccuracy(model, softmax):
    accuracy = model.Accuracy([softmax, 'label'], "accuracy")
    return accuracy


def VGG_Net(model,loss_scale):
    #----- 3 x 224 x 224 --> 64 x 224 x 224 -----#
    conv1_1 = brew.conv(model, 'data', 'conv1_1', 3, 64, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu1_1 = brew.relu(model, conv1_1, 'relu1_1')
    #----- 64 x 224 x 224 --> 64 x 224 x 224 -----#
    conv1_2 = brew.conv(model, relu1_1, 'conv1_2', 64, 64, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu1_2 = brew.relu(model, conv1_2, 'relu1_2')
    #----- 64 x 224 x 224 --> 64 x 112 x 112 -----#
    pool1 = brew.max_pool(model, relu1_2, 'pool1', kernel=2, stride=2)

    #----- 64 x 112 x 112 --> 128 x 112 x 112 -----#
    conv2_1 = brew.conv(model, pool1, 'conv2_1', 64, 128, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu2_1 = brew.relu(model, conv2_1, 'relu2_1')
    #----- 128 x 112 x 112 --> 128 x 112 x 112 -----#
    conv2_2 = brew.conv(model, relu2_1, 'conv2_2', 128, 128, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu2_2 = brew.relu(model, conv2_2, 'relu2_2')
    #----- 128 x 112 x 112 --> 128 x 56 x 56 -----#
    pool2 = brew.max_pool(model, relu2_2, 'pool2', kernel=2, stride=2)

    #----- 128 x 56 x 56 --> 256 x 56 x 56 -----#
    conv3_1 = brew.conv(model, pool2, 'conv3_1', 128, 256, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu3_1 = brew.relu(model, conv3_1, 'relu3_1')
    #----- 256 x 56 x 56 --> 256 x 56 x 56 -----#
    conv3_2 = brew.conv(model, relu3_1, 'conv3_2', 256, 256, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu3_2 = brew.relu(model, conv3_2, 'relu3_2')
    #----- 256 x 56 x 56 --> 256 x 56 x 56 -----#
    conv3_3 = brew.conv(model, relu3_2, 'conv3_3', 256, 256, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu3_3 = brew.relu(model, conv3_3, 'relu3_3')
    #----- 256 x 56 x 56 --> 256 x 28 x 28 -----#
    pool3 = brew.max_pool(model, relu3_3, 'pool3', kernel=2, stride=2)

    #----- 256 x 28 x 28 --> 512 x 28 x 28 -----#
    conv4_1 = brew.conv(model, pool3, 'conv4_1', 256, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu4_1 = brew.relu(model, conv4_1, 'relu4_1')
    #----- 512 x 28 x 28 --> 512 x 28 x 28 -----#
    conv4_2 = brew.conv(model, relu4_1, 'conv4_2', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu4_2 = brew.relu(model, conv4_2, 'relu4_2')
    #----- 512 x 28 x 28 --> 512 x 28 x 28 -----#
    conv4_3 = brew.conv(model, relu4_2, 'conv4_3', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu4_3 = brew.relu(model, conv4_3, 'relu4_3')
    #----- 512 x 28 x 28 --> 512 x 14 x 14 -----#
    pool4 = brew.max_pool(model, relu4_3, 'pool4', kernel=2, stride=2)

    #----- 512 x 14 x 14 --> 512 x 14 x 14 -----#
    conv5_1 = brew.conv(model, pool4, 'conv5_1', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu5_1 = brew.relu(model, conv5_1, 'relu5_1')
    #----- 512 x 14 x 14 --> 512 x 14 x 14 -----#
    conv5_2 = brew.conv(model, relu5_1, 'conv5_2', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu5_2 = brew.relu(model, conv5_2, 'relu5_2')
    #----- 512 x 14 x 14 --> 512 x 14 x 14 -----#
    conv5_3 = brew.conv(model, relu5_2, 'conv5_3', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
    relu5_3 = brew.relu(model, conv5_3, 'relu5_3')
    #----- 512 x 14 x 14 --> 512 x 7 x 7 -----#
    pool5 = brew.max_pool(model, relu5_3, 'pool5', kernel=2, stride=2)

    fc6 = brew.fc(model, pool5, 'fc6', 25088, 4096)
    relu6 = brew.relu(model, fc6,'relu6')

    drop6 = brew.dropout(model, relu6, 'drop6', ratio=0.5, is_test=0)

    fc7 = brew.fc(model, drop6, 'fc7', 4096, 4096)
    relu7 = brew.relu(model, fc7,'relu7')
    drop7 = brew.dropout(model, relu7,'drop7',ratio=0.5,is_test=0)

    fc8 = brew.fc(model, drop7, 'fc8', 4096, 2622)
    softmax = brew.softmax(model, fc8, 'softmax')
        
    xent = model.LabelCrossEntropy([softmax, 'label'], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax)
    loss = model.Scale(loss, "loss", scale=loss_scale)       

    return [loss]

def AddTrainingOperators(model):
    """
    opt = optimizer.build_sgd(model, base_learning_rate=1e-5, policy="step", stepsize=1, gamma=0.999, momentum=0.9)
#    model.AddWeightDecay(1e-4)
    """
#    brew.add_weight_decay(model, 1e-4)
    ITER = brew.iter(model, "iter")
#    ITER = model.Iter("iter")
    LR = model.LearningRate(ITER, "LR", base_lr=0.01, policy="step", stepsize=1, gamma=0.999, momentum=0.9 )
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    for param in model.GetParams():
        param_grad = model.param_to_grad[param]
        param_momentum = model.param_init_net.ConstantFill(
            [param], param + '_momentum', value=0.0
        )
        model.net.MomentumSGDUpdate(
            [param_grad, param_momentum, LR, param],
            [param_grad, param_momentum, param],
            momentum=0.9,
            nesterov=1,
        )
    return

def add_image_input_ops(model):
    # utilize the ImageInput operator to prep the images
    data, label = brew.image_input(
        model,
        reader,
        ["data", "label"],
        batch_size=batch_size,
        # mean: to remove color values that are common
        mean=128.,
        # std is going to be modified randomly to influence the mean subtraction
        std=128.,
        # scale to rescale each image to a common size
        scale=256,
        # crop to the square each image to exact dimensions
        crop=image_size,
        # not running in test mode
        is_test=False,
        # mirroring of the images will occur randomly
        mirror=1,
        use_caffe_datum=False,
    )
    # prevent back-propagation: optional performance improvement; may not be observable at small scale
    data = model.net.StopGradient(data, data)

def accuracy(model):
    accuracy = []
    for device in model._devices:
        accuracy.append(
            np.asscalar(workspace.FetchBlob("gpu_{}/accuracy".format(device))))
    return np.average(accuracy)

# Create ModelHelper object
train_arg_scope = {
    'order': 'NCHW',
    'use_gpu_engine': True,
    }
train_model = model_helper.ModelHelper(
    name="VGG", arg_scope=train_arg_scope
    )
reader = train_model.CreateDB(
    "train_reader",
    db=train_data,
    db_type="lmdb")

dpm.Parallelize_GPU(
    train_model,
    input_builder_fun=add_image_input_ops,
    forward_pass_builder_fun=VGG_Net,
    param_update_builder_fun=AddTrainingOperators,
    devices=gpus,
    optimize_gradient_memory=True,
    #cpu_device=True,
    )
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net)
print(train_model.net.Proto())

for epoch in range(num_epochs):
    num_iters = int(epoch_size / batch_size)
    for iter in range(num_iters):
        t1 = time.time()
        workspace.RunNet(train_model.net.Proto().name)
        t2 = time.time()
        dt = t2 - t1
        loss = []
        accuracy = []
        for device in train_model._devices:
            loss.append(workspace.FetchBlob("gpu_"+str(device)+"/loss"))
            accuracy.append(workspace.FetchBlob("gpu_"+str(device)+"/accuracy"))
        print('Loss = ' + str(np.average(loss)) + " --- Accuracy = " + str(np.average(accuracy)))

        """print((
            "Finished iteration {:>" + str(len(str(num_iters))) + "}/{}" +
            " (epoch {:>" + str(len(str(num_epochs))) + "}/{})" + 
            " ({:.2f} images/sec)").
            format(iter+1, num_iters, epoch+1, num_epochs, batch_size/dt))
		
        # Get the average accuracy for the training model
        train_accuracy = accuracy(train_model)
        print("Train accuracy: {:.3f}".format(train_accuracy))
        """