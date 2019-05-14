import numpy as np
import os
from PIL import Image
import scipy.misc
import tensorflow as tf

import UNet
from pipeline import build_batch

import copy

model = UNet.UNet(pretrained=True)
blur_dir = '../data/labelled_blurry'
sharp_dir = '../data/labelled_ground_truth'
predict_dir = '../data/predictions'
blurry_files = os.listdir(blur_dir)[0:10]
blurry_inputs = []

for filename in os.listdir(predict_dir):
    os.unlink(os.path.join(predict_dir, filename))

for filename in blurry_files:
    with Image.open(os.path.join(blur_dir, filename), 'r') as blurry:
        blurry_inputs.append(np.array(blurry))
        blurry.save(os.path.join(predict_dir,
            'blurry_{0}'.format(filename)),
            'PNG')
    with Image.open(os.path.join(sharp_dir, filename), 'r') as sharp:
        sharp.save(os.path.join(predict_dir,
            'sharp_{0}'.format(filename)),
            'PNG')

batch_generator = build_batch(blur_dir, sharp_dir, 10, resolution=128)
blurry_inputs, sharp_batch = next(batch_generator)
blurry_inputs = (np.array(blurry_inputs) / 127.5)  - 1.0
np.save('../data/blurry.npy', blurry_inputs)
# b&w Hack
# blurry_inputs = np.reshape(blurry_inputs, (*blurry_inputs.shape, 1))
print(blurry_inputs[0])
predictions = model.predict(blurry_inputs)
print('prediction:')
predictions = (predictions + 1.0) / 2.0
np.save('../data/prediction.npy', predictions)


for index,item in enumerate(predictions):
    # old_item = copy.deepcopy(item)
    # holder1, holder2 = item[:, :, 0], item[:, :, 2]
    # item[:, :, 0], item[:, :, 2] = holder2, holder1
    # (res, res, 3) if color else (res, res)
    print(item.shape)
    scipy.misc.imsave(os.path.join(predict_dir,
                    'prediction_{0}'.format(blurry_files[index])),
                    item)
    # with scipy.misc.toimage(item.astype(np.uint8)) as image:
    #     image.save(os.path.join(predict_dir,
    #                 'prediction_{0}'.format(blurry_files[index])),
    #                 'PNG')

