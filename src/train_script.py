import pipeline
import UNet

model = UNet.UNet()
train_steps = 200
blur_dir = '../data/labelled_patches/blurred'
truth_dir = '../data/labelled_patches/sharp'

pipeline.train_model(model, train_steps, blur_dir, truth_dir)