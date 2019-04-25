import pipeline
import UNet

model = UNet.UNet(learning_rate=1e-5)
train_steps = 100
blur_dir = '../data/labelled_patches/blurred'
truth_dir = '../data/labelled_patches/sharp'

pipeline.train_model(model, train_steps, blur_dir, truth_dir)
