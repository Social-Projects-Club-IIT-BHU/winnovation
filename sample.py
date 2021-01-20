import pandas as pd
import numpy as np
import cv2
import os
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt


seq = iaa.Sequential([
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-0.2*45, 0.2*45, ),
        shear=(-0.1*45, 0.1*45),
        mode = 'edge'
    ),
    iaa.size.Resize((64,64))
], random_order=True) # apply augmenters in random order



cap = cv2.VideoCapture('sample_videos/00414.mp4') 
out = cv2.VideoWriter('/media/amshra267/sa44/winnovation/sample_videos/augmented_video2.mp4', 0x7634706d, 20.0, (64,64))

augmented_images = []
i = 0
while(True):

    # Capture the video frame 
    # by frame 
    ret, frame = cap.read()
    if ret == False:
        break

  #  img = seq(images=[frame])
    cv2.imshow('original', cv2.resize(frame,(64,64)))
   # cv2.imshow('augmentation', np.array(img).reshape(64,64,3))
    augmented_images.append(frame)
  #  out.write(np.array(img).reshape(64,64,3))
    cv2.waitKey(10)

augseq_det = seq.to_deterministic()
augmented_images = [augseq_det.augment_image(frame).reshape(64,64,3) for frame in augmented_images]

for i in augmented_images:
    out.write(i)
cap.release()
out.release
cv2.destroyAllWindows()