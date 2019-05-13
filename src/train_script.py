import pipeline
import UNet

model = UNet.UNet(learning_rate=1e-3, pretrained=False)
train_steps = 1001
blur_dir = '../data/labelled_blurry'
truth_dir = '../data/labelled_ground_truth'
vblur_dir = '../data/validation_blurry'
vtruth_dir = '../data/validation_ground_truth'

pipeline.train_model(model, train_steps, blur_dir, truth_dir, vblur_dir, vtruth_dir,  resolution=128)
