from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace, model_helper, net_drawer, memonger, brew
from caffe2.python import data_parallel_model as dpm
from caffe2.python.models import resnet
from caffe2.proto import caffe2_pb2

import numpy as np
import time
import os
    
workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])

# This section checks if you have the training and testing databases
current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks')
data_folder = os.path.join(current_folder, 'tutorial_data', 'resnet_trainer')

# Train/test data
train_data_db = os.path.join(data_folder, "imagenet_cars_boats_train")
train_data_db_type = "lmdb"
# actually 640 cars and 640 boats = 1280
train_data_count = 1280
test_data_db = os.path.join(data_folder, "imagenet_cars_boats_val")
test_data_db_type = "lmdb"
# actually 48 cars and 48 boats = 96
test_data_count = 96

# Get the dataset if it is missing
def DownloadDataset(url, path):
    import requests, zipfile, StringIO
    print("Downloading {} ... ".format(url))
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall(path)
    print("Done downloading to {}!".format(path))

# Make the data folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
else:
    print("Data folder found at {}".format(data_folder))
# See if you already have to db, and if not, download it
if not os.path.exists(train_data_db):
    DownloadDataset("https://download.caffe2.ai/databases/resnet_trainer.zip", data_folder)

# Configure how you want to train the model and with how many GPUs
# This is set to use two GPUs in a single machine, but if you have more GPUs, extend the array [0, 1, 2, n]
gpus = [0]

# Batch size of 32 sums up to roughly 5GB of memory per device
batch_per_device = 32
total_batch_size = batch_per_device * len(gpus)

# This model discriminates between two labels: car or boat
num_labels = 2

# Initial learning rate (scale with total batch size)
base_learning_rate = 0.0004 * total_batch_size

# only intends to influence the learning rate after 10 epochs
stepsize = int(10 * train_data_count / total_batch_size)

# Weight decay (L2 regularization)
weight_decay = 1e-4

workspace.ResetWorkspace()

train_model = model_helper.ModelHelper(
    name="train",
)

reader = train_model.CreateDB(
    "train_reader",
    db=train_data_db,
    db_type=train_data_db_type,
)

def add_image_input_ops(model):
    # utilize the ImageInput operator to prep the images
    data, label = brew.image_input(
        model,
        reader,
        ["data", "label"],
        batch_size=batch_per_device,
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
        mirror=1
    )
    # prevent back-propagation: optional performance improvement; may not be observable at small scale
    data = model.net.StopGradient(data, data)

def create_resnet50_model_ops(model, loss_scale=1.0):
    # Creates a residual network
    [softmax, loss] = resnet.create_resnet50(
        model,
        "data",
        num_input_channels=3,
        num_labels=num_labels,
        label="label",
    )
    prefix = model.net.Proto().name
    loss = model.net.Scale(loss, prefix + "_loss", scale=loss_scale)
    brew.accuracy(model, [softmax, "label"], prefix + "_accuracy")
    return [loss]

def add_parameter_update_ops(model):
    brew.add_weight_decay(model, weight_decay)
    iter = brew.iter(model, "iter")
    lr = model.net.LearningRate(
        [iter],
        "lr",
        base_lr=base_learning_rate,
        policy="step",
        stepsize=stepsize,
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

"""
def optimize_gradient_memory(model, loss):
    model.net._net = memonger.share_grad_blobs(
        model.net,
        loss,
        set(model.param_to_grad.values()),
        namescope="imonaboat",
        share_activations=False,
        )

device_opt = core.DeviceOption(caffe2_pb2.HIP, gpus[0])
with core.NameScope("imonaboat"):
    with core.DeviceScope(device_opt):
        add_image_input_ops(train_model)
        losses = create_resnet50_model_ops(train_model)
        blobs_to_gradients = train_model.AddGradientOperators(losses)
        add_parameter_update_ops(train_model)
    optimize_gradient_memory(train_model, [blobs_to_gradients[losses[0]]])


workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite=True)

num_epochs = 1
for epoch in range(num_epochs):
    # Split up the images evenly: total images / batch size
    num_iters = int(train_data_count / total_batch_size)
    for iter in range(num_iters):
        # Stopwatch start!
        t1 = time.time()
        # Run this iteration!
        workspace.RunNet(train_model.net.Proto().name)
        t2 = time.time()
        dt = t2 - t1
        
        # Stopwatch stopped! How'd we do?
        print((
            "Finished iteration {:>" + str(len(str(num_iters))) + "}/{}" +
            " (epoch {:>" + str(len(str(num_epochs))) + "}/{})" + 
            " ({:.2f} images/sec)").
            format(iter+1, num_iters, epoch+1, num_epochs, total_batch_size/dt))
"""

dpm.Parallelize_GPU(
    train_model,
    input_builder_fun=add_image_input_ops,
    forward_pass_builder_fun=create_resnet50_model_ops,
    param_update_builder_fun=add_parameter_update_ops,
    devices=gpus,
    optimize_gradient_memory=True,
)

workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net)

test_model = model_helper.ModelHelper(
    name="test",
)

reader = test_model.CreateDB(
    "test_reader",
    db=test_data_db,
    db_type=test_data_db_type,
)

# Validation is parallelized across devices as well
dpm.Parallelize_GPU(
    test_model,
    input_builder_fun=add_image_input_ops,
    forward_pass_builder_fun=create_resnet50_model_ops,
    param_update_builder_fun=None,
    devices=gpus,
)

workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)

from caffe2.python import visualize
from matplotlib import pyplot as plt

def display_images_and_confidence():
    images = []
    confidences = []
    n = 16
    data = workspace.FetchBlob("gpu_0/data")
    label = workspace.FetchBlob("gpu_0/label")
    softmax = workspace.FetchBlob("gpu_0/softmax")
    for arr in zip(data[0:n], label[0:n], softmax[0:n]):
        # CHW to HWC, normalize to [0.0, 1.0], and BGR to RGB
        bgr = (arr[0].swapaxes(0, 1).swapaxes(1, 2) + 1.0) / 2.0
        rgb = bgr[...,::-1]
        images.append(rgb)
        confidences.append(arr[2][arr[1]])

    # Create grid for images
    fig, rows = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    plt.tight_layout(h_pad=2)

    # Display images and the models confidence in their label
    items = zip([ax for cols in rows for ax in cols], images, confidences)
    for (ax, image, confidence) in items:
        ax.imshow(image)
        if confidence >= 0.5:
            ax.set_title("RIGHT ({:.1f}%)".format(confidence * 100.0), color='green')
        else:
            ax.set_title("WRONG ({:.1f}%)".format(confidence * 100.0), color='red')

    plt.show()

    
def accuracy(model):
    accuracy = []
    prefix = model.net.Proto().name
    for device in model._devices:
        accuracy.append(
            np.asscalar(workspace.FetchBlob("gpu_{}/{}_accuracy".format(device, prefix))))
    return np.average(accuracy)

num_epochs = 6
for epoch in range(num_epochs):
    # Split up the images evenly: total images / batch size
    num_iters = int(train_data_count / total_batch_size)
    for iter in range(num_iters):
        # Stopwatch start!
        t1 = time.time()
        # Run this iteration!
        workspace.RunNet(train_model.net.Proto().name)
        t2 = time.time()
        dt = t2 - t1
        
        # Stopwatch stopped! How'd we do?
        print((
            "Finished iteration {:>" + str(len(str(num_iters))) + "}/{}" +
            " (epoch {:>" + str(len(str(num_epochs))) + "}/{})" + 
            " ({:.2f} images/sec)").
            format(iter+1, num_iters, epoch+1, num_epochs, total_batch_size/dt))
        
        # Get the average accuracy for the training model
        train_accuracy = accuracy(train_model)
    
    # Run the test model and assess accuracy
    test_accuracies = []
    for _ in range(test_data_count // total_batch_size):
        # Run the test model
        workspace.RunNet(test_model.net.Proto().name)
        test_accuracies.append(accuracy(test_model))
    test_accuracy = np.average(test_accuracies)

    print(
        "Train accuracy: {:.3f}, test accuracy: {:.3f}".
        format(train_accuracy, test_accuracy))
    
    # Output images with confidence scores as the caption
    #display_images_and_confidence()
