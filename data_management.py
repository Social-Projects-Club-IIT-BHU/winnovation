import numpy as np
import pandas as pd
import os
import glob
import cv2

class Data_preparation:
    '''
    Prerequisite for converting the data and format it into train, test, val
    '''

    def __init__(self, data_df = None):
        '''
        Constructor for calling the preparation clss

        Arguments:
           
           data_df - dataframe containing the paths and corresponding labels

           dest_directory - destination folder where we store the output frame level images of videos

        '''   

        self.df = data_df

    def division_paths(self, top_labels = 10, validation_ratio = 0.18, test_ratio = 0.12):
        top_lab = list(self.df['label'].value_counts()[0:top_labels].index)
        count_lab = list(self.df['label'].value_counts()[0:top_labels])

        for i, lab in enumerate(top_lab):
            train_len = int((1-(validation_ratio + test_ratio))*count_lab[i])          # train_portion is what remained from test and valid
            val_len = int(validation_ratio*count_lab[i])

            train_df = self.df[self.df['label'] == lab][0:train_len]
            val_df = self.df[self.df['label'] == lab][train_len: train_len + val_len]
            test_df = self.df[self.df['label'] == lab][train_len + val_len : ]
            
            # selecting those ones and concatenating them to make a dataframe

            if (i==0):
                train = train_df.reset_index(drop = True)
                val  = val_df.reset_index(drop = True)
                test = test_df.reset_index(drop = True)

            else:
                train = pd.concat([train, train_df], axis=0).reset_index(drop = True)
                val = pd.concat([val, val_df], axis=0).reset_index(drop = True)
                test = pd.concat([test, test_df], axis=0).reset_index(drop = True)
        

        return train, val, test 



    def frame_div(self, folder_path = 'Dataset'):    
         
        if not os.path.exists(folder_path):  # Dataset.
            os.makedirs(folder_path)

        
        
        if not os.path.exists(os.path.join(folder_path, 'valid')):  # valid
             os.makedirs(os.path.join(folder_path, 'valid'))

        if not os.path.exists(os.path.join(folder_path, 'test')):  # test
             os.makedirs(os.path.join(folder_path, 'test'))

        train, val, test = self.division_paths()

        self._set_individual_folder(data = train, parent_dir = folder_path, name = 'train')
        self._set_individual_folder(data = val, parent_dir = folder_path, name =  'valid')
        self._set_individual_folder(data = test, parent_dir = folder_path, name = 'test')
            
            

    def _set_individual_folder(self, data = None, parent_dir = None, name = None):

        if not os.path.exists(os.path.join(parent_dir, name)):  # Checking whether the given folder exist or not, if not make it.
             os.makedirs(os.path.join(parent_dir, name))
        
        for i in range(len(data)):
            path = data.iloc[i]['video_path']
            label = data.iloc[i]['label']

            if not os.path.exists(os.path.join(parent_dir, name, label)):  # checking particular label directory is made or not
                os.makedirs(os.path.join(parent_dir, name, label))
            
            dest_folder = os.path.join(parent_dir, name, label,os.path.basename(path).split('.')[0])
            os.makedirs(dest_folder)  # making directory for each particular video
            
            self._videowriter(video_path = path, save_folder = dest_folder, format = 'png')


   

    def _videowriter(self, video_path  = None, save_folder = None, format = None):

        # define a video capture object 
        
        cap = cv2.VideoCapture(video_path) 
        
        i = 0
        while(True):

            # Capture the video frame 
            # by frame 
            ret, frame = cap.read()
            if ret == False:
                break
            i +=1
            if format == 'png':   # images saved in png format
                cv2.imwrite(os.path.join(save_folder, 'img'+str(i)+'.png'),frame)
            else:
                cv2.imwrite(os.path.join(save_folder, 'img'+str(i)+'.jpg'),frame)

        
        cap.release()
        cv2.destroyAllWindows()
        

    