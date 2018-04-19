from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from matplotlib import pyplot as plt
import math
import numpy as np
import os
import lmdb
import shutil
from imageio import imread
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
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
core.GlobalInit(['caffe2', '--caffe2_log_level=-50000000', '--caffe2_gpu_memory_tracking=0'])
print("Necessities imported!")

import requests
import tarfile

# Set paths and variables
# data_folder is where the data is downloaded and unpacked
data_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10')
# root_folder is where checkpoint files and .pb model definition files will be outputted
root_folder = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_files', 'tutorial_cifar10')

url = "http://pjreddie.com/media/files/cifar.tgz"   # url to data
filename = url.split("/")[-1]                       # download file name
download_path = os.path.join(data_folder, filename) # path to extract data to

# Create data_folder if not already there
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

# If data does not already exist, download and extract
if not os.path.exists(download_path.strip('.tgz')):
    # Download data
    r = requests.get(url, stream=True)
    print("Downloading... {} to {}".format(url, download_path))
    open(download_path, 'wb').write(r.content)
    print("Finished downloading...")

    # Unpack images from tgz file
    print('Extracting images from tarball...')
    tar = tarfile.open(download_path, 'r')
    for item in tar:
        tar.extract(item, data_folder)
    print("Completed download and extraction!")
    
else:
    print("Image directory already exists. Moving on...")

# Paths to train and test directories
training_dir_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'cifar', 'train')
testing_dir_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'cifar', 'test')

# Paths to label files
training_labels_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'training_dictionary.txt')
validation_labels_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'validation_dictionary.txt')
testing_labels_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'testing_dictionary.txt')

# Paths to LMDBs
training_lmdb_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'training_lmdb')
validation_lmdb_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'validation_lmdb')
testing_lmdb_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'testing_lmdb')

# Path to labels.txt
labels_path = os.path.join(os.path.expanduser('~'), 'caffe2_notebooks', 'tutorial_data', 'cifar10', 'cifar', 'labels.txt')

# Open label file handler
labels_handler = open(labels_path, "r")

# Create classes dictionary to map string labels to integer labels
classes = {}
i = 0
lines = labels_handler.readlines()
for line in sorted(lines):
    line = line.rstrip()
    classes[line] = i
    i += 1
labels_handler.close()

print("classes:", classes)

from random import shuffle

# Open file handlers
training_labels_handler = open(training_labels_path, "w")
validation_labels_handler = open(validation_labels_path, "w")
testing_labels_handler = open(testing_labels_path, "w")

import glob

# Create training, validation, and testing label files
i = 0
validation_count = 6000
imgs = glob.glob(training_dir_path + '/*.png')  # read all training images into array
shuffle(imgs)  # shuffle array
for img in imgs:
    # Write first 6,000 image paths, followed by their integer label, to the validation label files
    if i < validation_count:
        validation_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
    # Write the remaining to the training label files
    else:
        training_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
    i += 1
print("Finished writing training and validation label files")

# Write our testing label files using the testing images
for img in glob.glob(testing_dir_path + '/*.png'):
    testing_labels_handler.write(img + ' ' + str(classes[img.split('_')[-1].split('.')[0]]) + '\n')
print("Finished writing testing label files")

# Close file handlers
training_labels_handler.close()
validation_labels_handler.close()
testing_labels_handler.close()

def write_lmdb(labels_file_path, lmdb_path):
    labels_handler = open(labels_file_path, "r")
    # Write to lmdb
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40
    print("LMDB_MAP_SIZE", LMDB_MAP_SIZE)
    env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE)

    with env.begin(write=True) as txn:
        count = 0
        for line in labels_handler.readlines():
            line = line.rstrip()
            im_path = line.split()[0]
            im_label = int(line.split()[1])
            
            # read in image (as RGB)
            img_data = imread(im_path).astype(np.float32)
            
            # convert to BGR
            img_data = img_data[:, :, (2, 1, 0)]
            
            # HWC -> CHW (N gets added in AddInput function)
            img_data = np.transpose(img_data, (2,0,1))
            
            # Create TensorProtos
            tensor_protos = caffe2_pb2.TensorProtos()
            img_tensor = tensor_protos.protos.add()
            img_tensor.dims.extend(img_data.shape)
            img_tensor.data_type = 1
            flatten_img = img_data.reshape(np.prod(img_data.shape))
            img_tensor.float_data.extend(flatten_img)
            label_tensor = tensor_protos.protos.add()
            label_tensor.data_type = 2
            label_tensor.int32_data.append(im_label)
            txn.put(
                '{}'.format(count).encode('ascii'),
                tensor_protos.SerializeToString()
            )
            if ((count % 1000 == 0)):
                print("Inserted {} rows".format(count))
            count = count + 1

    print("Inserted {} rows".format(count))
    print("\nLMDB saved at " + lmdb_path + "\n\n")
    labels_handler.close()

# Call function to write our LMDBs
if not os.path.exists(training_lmdb_path):
    print("Writing training LMDB")
    write_lmdb(training_labels_path, training_lmdb_path)
else:
    print(training_lmdb_path, "already exists!")
if not os.path.exists(validation_lmdb_path):
    print("Writing validation LMDB")
    write_lmdb(validation_labels_path, validation_lmdb_path)
else:
    print(validation_lmdb_path, "already exists!")
if not os.path.exists(testing_lmdb_path):
    print("Writing testing LMDB")
    write_lmdb(testing_labels_path, testing_lmdb_path)
else:
    print(testing_lmdb_path, "already exists!")

print(str(scope.CurrentDeviceScope()))
# Paths to the init & predict net output locations
init_net_out = 'cifar10_init_net.pb'
predict_net_out = 'cifar10_predict_net.pb'

# Dataset specific params
image_width = 32                # input image width
image_height = 32               # input image height
image_channels = 3              # input image channels (3 for RGB)
num_classes = 10                # number of image classes

# Training params
training_iters = 1000           # total training iterations
training_net_batch_size = 100   # batch size for training
validation_images = 6000        # total number of validation images
validation_interval = 100       # validate every <validation_interval> training iterations
checkpoint_iters = 1000         # output checkpoint db every <checkpoint_iters> iterations

# Create root_folder if not already there
if not os.path.isdir(root_folder):
    os.makedirs(root_folder)

# Resetting workspace with root_folder argument sets root_folder as working directory
workspace.ResetWorkspace(root_folder)

def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [],
        blob_out=["data_uint8", "label"],
        batch_size=batch_size,
        db=db,
        db_type=db_type,
    )
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label

def update_dims(height, width, kernel, stride, pad):
    new_height = math.ceil((height - kernel + 2*pad)/stride) + 1
    new_width = math.ceil((width - kernel + 2*pad)/stride) + 1
    return int(new_height), int(new_width)

def Add_Original_CIFAR10_Model(model, data, num_classes, image_height, image_width, image_channels):
    # Convolutional layer 1
    conv1 = brew.conv(model, data, 'conv1', dim_in=image_channels, dim_out=32, kernel=5, stride=1, pad=2)
    h,w = update_dims(height=image_height, width=image_width, kernel=5, stride=1, pad=2)
    # Pooling layer 1
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=3, stride=2, legacy_pad=3)
    h,w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)
    # ReLU layer 1
    relu1 = brew.relu(model, pool1, 'relu1')
    
    # Convolutional layer 2
    conv2 = brew.conv(model, relu1, 'conv2', dim_in=32, dim_out=32, kernel=5, stride=1, pad=2)
    h,w = update_dims(height=h, width=w, kernel=5, stride=1, pad=2)
    # ReLU layer 2
    relu2 = brew.relu(model, conv2, 'relu2')
    # Pooling layer 1
    pool2 = brew.average_pool(model, relu2, 'pool2', kernel=3, stride=2, legacy_pad=3)
    h,w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)
    
    # Convolutional layer 3
    conv3 = brew.conv(model, pool2, 'conv3', dim_in=32, dim_out=64, kernel=5, stride=1, pad=2)
    h,w = update_dims(height=h, width=w, kernel=5, stride=1, pad=2)
    # ReLU layer 3
    relu3 = brew.relu(model, conv3, 'relu3')
    # Pooling layer 3
    pool3 = brew.average_pool(model, relu3, 'pool3', kernel=3, stride=2, legacy_pad=3)
    h,w = update_dims(height=h, width=w, kernel=3, stride=2, pad=0)
    
    # Fully connected layers
    fc1 = brew.fc(model, pool3, 'fc1', dim_in=64*h*w, dim_out=64)
    fc2 = brew.fc(model, fc1, 'fc2', dim_in=64, dim_out=num_classes)
    
    # Softmax layer
    softmax = brew.softmax(model, fc2, 'softmax')
    return softmax

def AddTrainingOperators(model, softmax, label):
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # Compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # Use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # Use stochastic gradient descent as optimization function
    optimizer.build_sgd(
        model,
        base_learning_rate=0.01,
        policy="fixed",
        momentum=0.9,
        weight_decay=0.004
    )
"""
def AddTrainingOperators(model, softmax, label):
    #Adds training operators to the model.
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    ITER = brew.iter(model, "iter")
    LR = model.net.LearningRate(
        ITER, "LR", base_lr=-0.0001, policy="step", stepsize=1, gamma=0.999)
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.net.WeightedSum([param, ONE, param_grad, LR], param)
"""
def AddAccuracy(model, softmax, label):
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy
import datetime

# Create uniquely named directory under root_folder to output checkpoints to
unique_timestamp = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
checkpoint_dir = os.path.join(root_folder, unique_timestamp)
os.makedirs(checkpoint_dir)
print("Checkpoint output location: ", checkpoint_dir)

# Add checkpoints to a given model
def AddCheckpoints(model, checkpoint_iters, db_type):
    ITER = brew.iter(train_model, "iter")
    train_model.Checkpoint([ITER] + train_model.params, [], db=os.path.join(unique_timestamp, "cifar10_checkpoint_%05d.lmdb"), db_type="lmdb", every=checkpoint_iters)



device = core.DeviceOption(caffe2_pb2.HIP, 0)
with core.DeviceScope(device):

    #arg_scope = {"order": "NCHW"}
    arg_scope = {
            'order': "NCHW",
            'use_gpu_engine': True,
            'gpu_engine_exhaustive_search': False,
        }
    # TRAINING MODEL
    # Initialize with ModelHelper class
    train_model = model_helper.ModelHelper(
        name="train_net", arg_scope=arg_scope)
    #train_model.param_init_net.RunAllOnGPU()
    #train_model.net.RunAllOnGPU()
    # Add data layer from training_lmdb
    data, label = AddInput(
        train_model, batch_size=training_net_batch_size,
        db=training_lmdb_path,
        db_type='lmdb')
    # Add model definition, save return value to 'softmax' variable
    softmax = Add_Original_CIFAR10_Model(train_model, data, num_classes, image_height, image_width, image_channels)
    # Add training operators using the softmax output from the model
    AddTrainingOperators(train_model, softmax, label)
    # Add periodic checkpoint outputs to the model
    AddCheckpoints(train_model, checkpoint_iters, db_type="lmdb")

    # VALIDATION MODEL
    # Initialize with ModelHelper class without re-initializing params
    val_model = model_helper.ModelHelper(
        name="val_net", arg_scope=arg_scope, init_params=False)
    #val_model.param_init_net.RunAllOnGPU()
    #val_model.net.RunAllOnGPU()
    # Add data layer from validation_lmdb
    data, label = AddInput(
        val_model, batch_size=validation_images,
        db=validation_lmdb_path,
        db_type='lmdb')
    # Add model definition, save return value to 'softmax' variable
    softmax = Add_Original_CIFAR10_Model(val_model, data, num_classes, image_height, image_width, image_channels)
    # Add accuracy operator
    AddAccuracy(val_model, softmax, label)


    # DEPLOY MODEL
    # Initialize with ModelHelper class without re-initializing params
    deploy_model = model_helper.ModelHelper(
        name="deploy_net", arg_scope=arg_scope, init_params=False)
    #deploy_model.param_init_net.RunAllOnGPU()
    #deploy_model.net.RunAllOnGPU()
    # Add model definition, expect input blob called "data"
    Add_Original_CIFAR10_Model(deploy_model, "data", num_classes, image_height, image_width, image_channels)

    print("Training, Validation, and Deploy models all defined!")

    import math

    # Initialize and create the training network
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)
    # Initialize and create validation network
    workspace.RunNetOnce(val_model.param_init_net)
    workspace.CreateNet(val_model.net, overwrite=True)
    # Placeholder to track loss and validation accuracy
    loss = np.zeros(int(math.ceil(training_iters/validation_interval)))
    val_accuracy = np.zeros(int(math.ceil(training_iters/validation_interval)))
    val_count = 0
    iteration_list = np.zeros(int(math.ceil(training_iters/validation_interval)))

    # Now, we run the network (forward & backward pass)
    for i in range(training_iters):
        workspace.RunNet(train_model.net)
        # Validate every <validation_interval> training iterations
        if (i % validation_interval == 0):
            print("Training iter: ", i)
            loss[val_count] = workspace.FetchBlob('loss')
            workspace.RunNet(val_model.net)
            val_accuracy[val_count] = workspace.FetchBlob('accuracy')
            print("Loss: ", str(loss[val_count]))
            print("Validation accuracy: ", str(val_accuracy[val_count]) + "\n")
            iteration_list[val_count] = i
            val_count += 1

    plt.title("Training Loss vs. Validation Accuracy")
    plt.plot(iteration_list, loss, 'b')
    plt.plot(iteration_list, val_accuracy, 'r')
    plt.xlabel("Training iteration")
    plt.legend(('Loss', 'Validation Accuracy'), loc='upper right')
    #plt.show()

    workspace.RunNetOnce(deploy_model.param_init_net)
    workspace.CreateNet(deploy_model.net, overwrite=True)

    # Use mobile_exporter's Export function to acquire init_net and predict_net
    init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)

    # Locations of output files
    full_init_net_out = os.path.join(checkpoint_dir, init_net_out)
    full_predict_net_out = os.path.join(checkpoint_dir, predict_net_out)

    # Simply write the two nets to file
    with open(full_init_net_out, 'wb') as f:
        f.write(init_net.SerializeToString())
    with open(full_predict_net_out, 'wb') as f:
        f.write(predict_net.SerializeToString())
    print("Model saved as " + full_init_net_out + " and " + full_predict_net_out)
