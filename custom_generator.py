import pandas as pd
import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from collections import deque
import copy
from keras.utils import np_utils

class VideoDataGenerator:

   '''
      custom data generator for video augmentation

   '''

   def __init__( self,
               rotation_range=0,
               width_shift_range=0.0,
               height_shift_range=0.0,
               zoom_range=0.0,
               fill_mode="nearest",
              # horizontal_flip=False,
              # vertical_flip=False,
               rescale=None,
               base_path = None,
               temporal_length = 8,
               temporal_stride = 1,
               shape = [64,64],
               labels = 10
              ):

      '''
      Constructor function for calling video data generator 


      '''
      self.rotation = rotation_range 
      self.width_shift = width_shift_range
      self.height_shift = height_shift_range
      self.zoom = zoom_range
      self.fill_mode = fill_mode
    #  self.horizontal_flip = horizontal_flip
    #  self.vertical_flip = vertical_flip
      self.rescale = rescale
      self.temporal_length = temporal_length
      self.temporal_stride = temporal_stride
      self.base_path = base_path
      self.shape = shape
      self.labels = labels
      self._form_csv(data_path = os.path.join(self.path, 'train'))  # making csv for train data
      self._form_csv(data_path = os.path.join(self.path, 'valid'))  # making csv for train data
      self._form_csv(data_path = os.path.join(self.path, 'test'))  # making csv for train data
     # self._file_generator()




   def _form_csv(self, data_path = None):
      '''
      making a csv file for each video in the dataset 
      '''
      
      if not os.path.exists(os.path.join('csv_datatest', os.path.basename(data_path))):
         os.makedirs(os.path.join('csv_dataset', os.path.basename(data_path)))

      data_dir_list = os.listdir(data_path)

      for data_dir in data_dir_list:
         label = str(data_dir)
         video_list = os.listdir(os.path.join(data_dir_list,label))
         for vid in video_list:  # loop over all the videos
            data_df = pd.DataFrame(columns = ['image_path', 'label'])
            img_list = os.listdir(os.path.join(data_path, label, vid))
            for img in img_list: # looping through all the images and then appending it to the dataframe
               img_path = os.path.join(data_path, label, vid, img)
               data_df.append({ 'image_path' : img_path, 'label' : label}, ignore_index = True)
                
            file_name = '{}_{}.csv'.format(data_dir,vid)
            data_df.to_csv(os.path.join('csv_dataset'), os.path.basename(data_path), file_name)  # saving video_by_video

   
   def file_generator(self,data_path,data_files):
      '''
      data_files - list of csv files to be read.
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

      return shuffle(samples,random_state=2)
    
   def preprocess_image(self,img):


      img = cv2.resize(img,(self.shape[0], self.shape[1]))
      img = img*self.rescale

      return img
    
   def data_generator(self,data, labels_map_dict = None, batch_size=10,shuffle=True):              
      """
      Yields the next training batch.
      data is an array [[img1_filename,img2_filename...,img8_filename],label1], [image2_filename,label2],...].
      """
      num_samples = len(data)
      if shuffle:
         data = self.shuffle_data(data)
      while True:   
         for offset in range(0, num_samples, batch_size):
            #print ('startring index: ', offset) 
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            # Initialise X_train and y_train arrays for this batch
            x_train = []
            y_train = []
            # For each example
            for batch_sample in batch_samples:
               # Load image (X)
               x = batch_sample[0]
               y = batch_sample[1]
               temp_data_list = []
               for img in x:
                  try:
                     img = cv2.imread(img)
                     #apply any kind of preprocessing here
                     #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                     img = self.preprocess_image(img)
                     temp_data_list.append(img)

                  except Exception as e:
                     print (e)
                     print ('error reading file: ',img)  

               # Read label (y)
               #label = label_names[y]
               # Add example to arrays
               x_train.append(temp_data_list)
               y_train.append(y)
   
            # Make sure they're numpy arrays (as opposed to lists)
            x_train = np.asarray(x_train)
            # this integer encoding is purely based on position, you can do this in other ways

            y_train = [labels_map_dict[label] for label in y_train]
            
            y_train = np.asarray(y_train)
            y_train = np_utils.to_categorical(y_train, self.labels)

            # The generator-y part: yield the next training batch            
            yield x_train, y_train

