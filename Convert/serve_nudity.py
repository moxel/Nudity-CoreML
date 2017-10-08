import caffe
import caffe.io
import numpy as np
import moxel


age_net = caffe.Classifier('deploy.prototxt', 'resnet_50_1by2_nsfw.caffemodel',
                           image_dims=(224, 224))

def predict(image):
    image = image.to_numpy_rgb()[:, :, :3]
    image[:,:,0] -= 104
    image[:,:,1] -= 117
    image[:,:,2] -= 123
    image[:,:,::-1] = image[:, :, :]
    image = np.array(image, dtype='float32')
    pred = age_net.predict([image])[0]
    result = pred.argmax()
    if result == 0:
        nude = 'no'
    else:
        nude = 'yes'
    return {
        'nude': nude
    }
