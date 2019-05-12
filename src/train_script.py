import pipeline
import UNet

model = UNet.UNet(learning_rate=1e-5, pretrained=False)
train_steps = 2000
blur_dir = '../data/labelled_blurry'
truth_dir = '../data/labelled_ground_truth'

pipeline.train_model(model, train_steps, blur_dir, truth_dir,  resolution=64)
