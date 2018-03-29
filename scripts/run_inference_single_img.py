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

# Function to crop the center cropX x cropY pixels from the input image
def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# Function to rescale the input image to the desired height and/or width. This function will preserve
#   the aspect ratio of the original image while making the image the correct scale so we can retrieve
#   a good center crop. This function is best used with center crop to resize any size input images into
#   specific sized images that our model can use.
def rescale(img, input_height, input_width):
    # Get original aspect ratio
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled



opts, args = getopt.getopt(sys.argv[1:], 'm:i:s:g:')

model = ''
inp_img = ''
img_size = 0
use_gpu = 0

for opt, arg in opts:
    if opt == '-m':
        model = arg
    elif opt == '-i':
        inp_img = arg
    elif opt == '-s':
        img_size = int(arg)
    elif opt == '-g':
        use_gpu = int(arg)

if model == '' or inp_img == '' or img_size == 0:
    print('Invalid command line argument')
    print('Usage:')
    print('-m <model directory> -i <input image> -s <input image size> -g [Use GPU option -- 0: CPU (default), 1: HIP, 2: MIOPEN]')
    print('Model directory must contain both the init_net.pb and predict_net.pb, and optionally ilsvrc_2012_mean.npy files')
    exit()

# codes - these help decypher the output and source from a list from ImageNet's object codes 
#    to provide an result like "tabby cat" or "lemon" depending on what's in the picture 
#   you submit to the CNN.
codes =  "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"

# set paths and variables from model choice and prep image
model = os.path.expanduser(model)

# mean can be 128 or custom based on the model
# gives better results to remove the colors found in all of the training images
mean_file = os.path.join(model, 'ilsvrc_2012_mean.npy')

if not os.path.exists(mean_file):
    print("No mean file found!")
    mean = 128
else:
    print ("Mean file found!")
    mean = np.load(mean_file).mean(1).mean(1)
    mean = mean[:, np.newaxis, np.newaxis]

# make sure all of the files are around...
INIT_NET = os.path.join(model, 'init_net.pb')
PREDICT_NET = os.path.join(model, 'predict_net.pb')

# Check to see if the files exist
if not os.path.exists(INIT_NET):
    print("WARNING: " + INIT_NET + " not found!")
else:
    if not os.path.exists(PREDICT_NET):
        print("WARNING: " + PREDICT_NET + " not found!")

# Load the image as a 32-bit float
#    Note: skimage.io.imread returns a HWC ordered RGB image of some size
img = skimage.img_as_float(skimage.io.imread(inp_img)).astype(np.float32)
#print("Original Image Shape: " , img.shape)

# Rescale the image to comply with our desired input size. This will not make the image 227x227
#    but it will make either the height or width 227 so we can get the ideal center crop.
img = rescale(img, img_size, img_size)
#print("Image Shape after rescaling: " , img.shape)
pyplot.figure()
pyplot.imshow(img)
pyplot.title('Rescaled image')

# Crop the center 227x227 pixels of the image so we can feed it to our model
img = crop_center(img, img_size, img_size)
#print("Image Shape after cropping: " , img.shape)
pyplot.figure()
pyplot.imshow(img)
pyplot.title('Center Cropped')

# switch to CHW (HWC --> CHW)
img = img.swapaxes(1, 2).swapaxes(0, 1)
#print("CHW Image Shape: " , img.shape)

#pyplot.figure()
for i in range(3):
    # For some reason, pyplot subplot follows Matlab's indexing
    # convention (starting with 1). Well, we'll just follow it...
    pyplot.subplot(1, 3, i+1)
    pyplot.imshow(img[i])
    pyplot.axis('off')
    pyplot.title('RGB channel %d' % (i+1))

# switch to BGR (RGB --> BGR)
img = img[(2, 1, 0), :, :]

# remove mean for better results
img = img * 255 - mean

# add batch size axis which completes the formation of the NCHW shaped input that we want
img = img[np.newaxis, :, :, :].astype(np.float32)

#print("NCHW image (ready to be used as input): ", img.shape)
pyplot.show()

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
### Debug code
"""
for b in workspace.Blobs():
    x = workspace.FetchBlob(b)
    if type(x) != str:
        print(str(b) + ': ' + str(x.shape))

"""
### End of debug code

print('Running net ' + workspace.GetNetName(net_def) + '...')

C.run_net(workspace.GetNetName(net_def), 1, False)

# Turn it into something we can play with and examine which is in a multi-dimensional array
results = workspace.FetchBlob('prob')
#print("results shape: ", results.shape)

# Quick way to get the top-1 prediction result
# Squeeze out the unnecessary axis. This returns a 1-D array of length 1000
preds = np.squeeze(results)
# Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
print("Prediction: ", curr_pred)
print("Confidence: ", curr_conf)

# the rest of this is digging through the results 
results = np.delete(results, 1)
index = 0
highest = 0
arr = np.empty((0,2), dtype=object)
arr[:,0] = int(10)
arr[:,1:] = float(10)
for i, r in enumerate(results):
    # imagenet index begins with 1!
    i=i+1
    arr = np.append(arr, np.array([[i,r]]), axis=0)
    if (r > highest):
        highest = r
        index = i 

# top N results
N = 5
topN = sorted(arr, key=lambda x: x[1], reverse=True)[:N]
print("Raw top {} results: {}".format(N,topN))

# Isolate the indexes of the top-N most likely classes
topN_inds = [int(x[0]) for x in topN]
print("Top {} classes in order: {}".format(N,topN_inds))

# Now we can grab the code list and create a class Look Up Table
response = urllib2.urlopen(codes)
class_LUT = []
for line in response:
    code, result = line.partition(":")[::2]
    code = code.strip()
    result = result.replace("'", "")
    if code.isdigit():
        class_LUT.append(result.split(",")[0][1:])
        
# For each of the top-N results, associate the integer result with an actual class
for n in topN:
    print("Model predicts '{}' with {}% confidence".format(class_LUT[int(n[0])],float("{0:.2f}".format(n[1]*100))))
