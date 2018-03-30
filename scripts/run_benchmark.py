from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot
import os, time, getopt, sys
from caffe2.python import core, workspace, models
import urllib2
import operator
import caffe2.python._import_c_extension as C

opts, args = getopt.getopt(sys.argv[1:], 'm:s:b:i:l:g:')

model = ''
img_size = 0
batch_size = 32
use_gpu = 0
num_iter = 50
per_layer_timing = False

for opt, arg in opts:
    if opt == '-m':
        model = arg
    elif opt == '-s':
        img_size = int(arg)
    elif opt == 'i':
    	num_iter = int(arg)
    elif opt == '-b':
        batch_size = int(arg)
    elif opt == '-g':
        use_gpu = int(arg)
    elif opt == '-l':
    	if int(arg) == 1:
    		per_layer_timing = True

if model == '' or img_size == 0:
    print('Invalid command line argument')
    print('Usage:')
    print('-m <model directory> -s <input image size> -i [num iterations. Default 50] -b [batch size. Default 32] -g [Use GPU option -- 0: CPU (default), 1: HIP, 2: MIOPEN] -l [0/1 Per layer timing. Default 0(OFF)]')
    print('Model directory must contain both the init_net.pb and predict_net.pb')
    exit()

model = os.path.expanduser(model)
INIT_NET = os.path.join(model, 'init_net.pb')
PREDICT_NET = os.path.join(model, 'predict_net.pb')

# Check to see if the files exist
if not os.path.exists(INIT_NET):
    print("WARNING: " + INIT_NET + " not found!")
else:
    if not os.path.exists(PREDICT_NET):
        print("WARNING: " + PREDICT_NET + " not found!")

img = np.random.rand(batch_size, 3, img_size, img_size).astype('f')
print(str(img.shape))

# Create GPU device option
device_opts = caffe2_pb2.DeviceOption()
if use_gpu == 0:
    device_opts.device_type = caffe2_pb2.CPU
    print('Running on CPU')
else:
    device_opts.device_type = caffe2_pb2.HIP
    device_opts.hip_gpu_id = 0
    print('Running on HIP')
    if use_gpu == 2:
        engine_list = ['MIOPEN', '']
        C.set_global_engine_pref({caffe2_pb2.HIP : engine_list})
        print('Using MIOPEN')

C.feed_blob('data', img, device_opts.SerializeToString())

init_def = caffe2_pb2.NetDef()
with open(INIT_NET, 'rb') as f:
    init_def.ParseFromString(f.read())
    init_def.device_option.CopyFrom(device_opts)
    C.run_net_once(init_def.SerializeToString())

net_def = caffe2_pb2.NetDef()
with open(PREDICT_NET, 'rb') as f:
    net_def.ParseFromString(f.read())
    net_def.device_option.CopyFrom(device_opts)
    C.create_net(net_def.SerializeToString())

C.feed_blob('data', img, device_opts.SerializeToString())
C.benchmark_net(workspace.GetNetName(net_def), 10, num_iter, per_layer_timing)