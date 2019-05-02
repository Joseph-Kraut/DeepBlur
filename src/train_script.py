import pipeline
import UNet

model = UNet.UNet(learning_rate=1e-5)
train_steps = 5000
blur_dir = '../data/labelled_blurry'
truth_dir = '../data/labelled_ground_truth'

pipeline.train_model(model, train_steps, blur_dir, truth_dir, resolution=192)
