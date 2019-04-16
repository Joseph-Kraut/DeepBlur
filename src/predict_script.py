import numpy as np
import os
from PIL import Image
import scipy.misc
import tensorflow as tf

import UNet

model = UNet.UNet(pretrained=True)
blur_dir = '../data/labelled_patches/blurred'
predict_dir = '../data/predictions'
blurry_files = os.listdir(blur_dir)[0:10]
blurry_inputs = []

for filename in os.listdir(predict_dir):
    os.unlink(os.path.join(predict_dir, filename))

for filename in blurry_files:
    with Image.open(os.path.join(blur_dir, filename), 'r') as blurry:
        blurry_inputs.append(np.array(blurry))
        blurry.save(os.path.join(predict_dir,
            '{0}'.format(filename)),
            'PNG')

blurry_inputs = np.array(blurry_inputs)
blurry_inputs = np.reshape(blurry_inputs, (*blurry_inputs.shape, 1))
print(blurry_inputs)
predictions = model.predict(np.array(blurry_inputs))
print(predictions)

makeint = lambda x: 0 if x < 0 else int(x)
vmakeint = np.vectorize(makeint)
for index,item in enumerate(predictions):
    item = vmakeint(item)
    item = np.reshape(item.astype(np.uint8), (300,300))
    print(item.astype(np.uint8))
    with scipy.misc.toimage(item.astype(np.uint8)) as image:
        image.save(os.path.join(predict_dir,
                    'prediction_{0}'.format(blurry_files[index])),
                    'PNG')

