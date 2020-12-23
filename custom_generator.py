import pandas as pd
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from collections import deque
import copy
from keras.utils import np_utils
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from concurrent.futures import ThreadPoolExecutor

class VideoDataGenerator:

   '''
      custom data generator for video augmentation

   '''

   def __init__( self,
               rotation_range=0.2,
               shear_range = 0.2,
               width_shift_range = 0.2,
               height_shift_range = 0.2,
               rescale=1./255,
               base_path = None,
               temporal_length = 6,
               temporal_stride = 1,
               shape = (64,64),
               labels = 5
              ):

      '''
      Constructor function for calling video data generator 

      Arguments:

      rotation_range - degree upto which image can rotate, in between (0,1) (default = 0.0),
      width_shift_range - amount of shiftimg in horizontal direction, in between (0,1) (default = 0.0),
      height_shift_range - amount of shiftimg in vertical direction, in between (0,1) (default = 0.0),
      zoom_range - amount of zooming, in between (0,1) (default = 0.0),
      fill_mode - interploation method (deafult = "nearest"),
      rescale - rescaling pixels according to multiplication by this(default = 1./255)
      base_path - root directory path inside which images are present, (default = None),
      temporal_length - No. of frames to be taken per video sample, (default = 8)
      temporal_stride - tmporal strides across each sample videos , (default = 1),
      shape  - width and height of each frame, (default = [64,64]),
      labels = 10

      '''
      self.rotation = rotation_range 
      self.width_shift = width_shift_range
      self.height_shift = height_shift_range
      self.shear_range = shear_range
      self.rescale = rescale
      self.temporal_length = temporal_length
      self.temporal_stride = temporal_stride
      self.base_path = base_path
      self.shape = shape
      self.labels = labels




   def csv_maker(self):
      self._form_csv(data_path = os.path.join(self.base_path, 'train'))  # making csv for train data
      self._form_csv(data_path = os.path.join(self.base_path, 'valid'))  # making csv for train data
      self._form_csv(data_path = os.path.join(self.base_path, 'test'))  # making csv for train data
      
   def _form_csv(self, data_path = None):
      '''
      making a csv file for each video in the dataset 

      Arguments:
      data_path = path of the folder from which csv is obtained

      '''
      
      if not os.path.exists(os.path.join('csv_dataset', os.path.basename(data_path))):
         os.makedirs(os.path.join('csv_dataset', os.path.basename(data_path)))

      data_dir_list = os.listdir(data_path)

      for data_dir in data_dir_list:
         label = str(data_dir)
         video_list = os.listdir(os.path.join(data_path,label))
         for vid in video_list:  # loop over all the videos
            data_df = pd.DataFrame(columns = ['image_path', 'label'])
            img_list = os.listdir(os.path.join(data_path, label, vid))
            for img in img_list: # looping through all the images and then appending it to the dataframe
               img_path = os.path.join(data_path, label, vid, img)
               data_df = data_df.append({ 'image_path' : img_path, 'label' : label}, ignore_index = True)
                
            file_name = '{}_{}.csv'.format(data_dir,vid)
            data_df.to_csv(os.path.join('csv_dataset', os.path.basename(data_path), file_name))  # saving video_by_video

   
   def file_generator(self,data_path,data_files):
      '''
      Function for making a file path generator for samples of videos according to temporal length and stride

      Argument - 
      data_files - list of csv files to be read.
      data_path - path of the particular image data folder
      
      yields:

      appropriate length frame samples and corresponding label
      '''
      for f in data_files:

         tmp_df = pd.read_csv(os.path.join(data_path,f))
         label_list = list(tmp_df['label'])
         total_images = len(label_list) 
         if total_images>=self.temporal_length:
            num_samples = int((total_images-self.temporal_length)/self.temporal_stride)+1
            print ('num of samples from vid seq-{}: {}'.format(f,num_samples))
            img_list = list(tmp_df['image_path'])
         else:
            print ('num of frames is less than temporal length; hence discarding this file-{}'.format(f))
            continue
         
         samples = deque()
         samp_count=0
         for img in img_list:
            samples.append(img)
            if len(samples)==self.temporal_length:
               samples_c=copy.deepcopy(samples)
               samp_count+=1
               for _ in range(self.temporal_stride):
                  samples.popleft() 
               yield samples_c,label_list[0]

   def load_samples(self,root_path = 'csv_dataset', data_cat=None):

      '''
      Function for loading all the samples path in a list

      Arguments:
      
      root_path - path for storing that category data samples
      data_cat - type of data  ('train', 'test', 'valid')
      
      Returns:
      
      list containg paths for frames of each sample and corresponding label
      '''

      data_path = os.path.join(root_path, data_cat)
      csv_data_files = os.listdir(data_path)
      file_gen = self.file_generator(data_path,csv_data_files)
      iterator = True
      data_list = []
      while iterator:
         try:
            x,y = next(file_gen)
            x=list(x)
            data_list.append([x,y])
         except Exception as e:
            print ('the exception: ',e)
            iterator = False
            print ('end of file generator')
      return data_list
    
   def shuffle_data(self, samples):

      '''
      data shuffling
      '''

      return shuffle(samples,random_state=2)


   def preprocess_video(self):
      
      '''
      This function contains the sequence of action to be done
      '''

      return iaa.Sequential([
         iaa.Crop(percent=(0, 0.1)), # random crops
         # Small gaussian blur with random sigma between 0 and 0.5.
         # But we only blur about 50% of all images.
         iaa.Sometimes(
            0.7,
            iaa.GaussianBlur(sigma=(0, 0.5))
         ),
         # Apply affine transformations to each image.
         # Scale/zoom them, translate/move them, rotate them and shear them.
         iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-self.width_shift, self.width_shift), "y": (-self.height_shift, self.height_shift)},
            rotate=(-self.shear_range*90, self.shear_range*90),
            shear=(-self.rotation*45, self.rotation*45),
            mode = 'edge'
         ),
      ], random_order=True) # apply augmenters in random order
   
   def batch_preparator(self, batch_samples, preprocess):
      
      x_train = []
      y_train = []
      for batch_sample in batch_samples:
         # Load image (X)
         x = batch_sample[0]
         y = batch_sample[1]
         temp_data_list = []
         for img in x:
            try:
               img = cv2.imread(img)
               img = cv2.resize(img, self.shape).astype('float32')
               img *=self.rescale
               temp_data_list.append(img) # appending all the images one by one

            except Exception as e:
               print (e)
               print ('error reading file: ',img)

         if preprocess: # if processing is true
            seq = self.preprocess_video()
            det = seq.to_deterministic()
            temp_data_list =  [det.augment_image(frame).reshape(self.shape[0],self.shape[1],3) for frame in temp_data_list]  # Augmenting and preprocessing each frame of a video in same way
         x_train.append(temp_data_list)
         y_train.append(y)
      
      return x_train, y_train   

    
   def flow(self,data, labels_map_dict = None, batch_size=10,shuffle=True, preprocessing = True):              
      """
      Yields the next training batch.
      data is an array [[img1_filename,img2_filename...,upto frames mentioned using temporal length],label1], [image2_filename,label2],...].
      """

      num_samples = len(data)
      
      if shuffle:
         data = self.shuffle_data(data)
      while True:
         with ThreadPoolExecutor(max_workers=16) as pool:   
            for offset in range(0, num_samples, batch_size):
               # Get the samples you'll use in this batch
               batch_samples = data[offset:offset+batch_size]
               # Initialise x_train and y_train arrays for this batch

               # For each example
               
               x_train, y_train = pool.submit(self.batch_preparator, batch_samples, preprocessing).result()
               # Make sure they're numpy arrays (as opposed to lists)
               x_train = np.asarray(x_train)

               # mapping labels
               y_train = [labels_map_dict[label] for label in y_train]
               
               y_train = np.asarray(y_train)
               y_train = np_utils.to_categorical(y_train, self.labels) # one hot encoding labels

               # The generator-y part: yield the next training batch            
               yield x_train, y_train

