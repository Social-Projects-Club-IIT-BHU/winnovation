import numpy as np
import pandas as pd
import os
import cv2

class Data_preparation:
    '''
    Prerequisite for converting the data and divide it into train, test, val
    '''

    def __init__(self, data_df = None, top_labels = 10):
        '''
        Constructor for calling the preparation clss

        Arguments:
           
           data_df - dataframe containing the paths and corresponding labels
           top_labels - Labels to be used for data preparation, (default = 10)
        '''   

        self.df = data_df
        self.top_labels = top_labels

    def _division_paths(self, validation_ratio = 0.18, test_ratio = 0.12):
        
        '''
        Function defining the division of data among train, test and valid

        Arguments:

        validation_ratio - for setting percentage validation data
        test_ratio - for setting percentage validation data
        top_labels - no. of labels for divison

        default values are mentioned above

        '''

        top_lab = list(self.df['label'].value_counts()[0:self.top_labels].index)
        count_lab = list(self.df['label'].value_counts()[0:self.top_labels])
        print(top_lab)

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



    def folder_div(self, folder_path = 'Dataset'):   

        '''
         function used for dividing the data folders into sub directories

         Arguments :

         top_labels - no. of labels on which we train our data (default = 10)

         folder_path - path where data is stored (default = 'Dataset')

        '''
         
        if not os.path.exists(folder_path):  # Dataset folder 
            os.makedirs(folder_path)


        train, val, test = self._division_paths()

        self._set_individual_folder(data = train, parent_dir = folder_path, name = 'train')
        self._set_individual_folder(data = val, parent_dir = folder_path, name =  'valid')
        self._set_individual_folder(data = test, parent_dir = folder_path, name = 'test')
            
            

    def _set_individual_folder(self, data = None, parent_dir = None, name = None):

        '''

        function for managing the individual folder directories

        Arguments:

        data - dataframe for managing that folder
        parent_dir - parent directory of that folder
        name - name of the folder

        '''

        if not os.path.exists(os.path.join(parent_dir, name)):  # Checking whether the given folder exist or not, if not make it.
             os.makedirs(os.path.join(parent_dir, name))
        
        for i in range(len(data)):
            path = data.iloc[i]['video_path']
            label = data.iloc[i]['label']

            if not os.path.exists(os.path.join(parent_dir, name, label)):  # checking particular label directory is made or not
                os.makedirs(os.path.join(parent_dir, name, label))
            
            dest_folder = os.path.join(parent_dir, name, label, os.path.basename(path).split('.')[0])
            os.makedirs(dest_folder)  # making directory for each particular video
            
            self._videowriter(video_path = path, save_folder = dest_folder, format = 'png')


   

    def _videowriter(self, video_path  = None, save_folder = None, format = 'png'):

        '''
        Function for diving a video into frames and store them one by one

        Arguments:

        video_path - Path of the video
        save_folder - folder where frames need to be save
        format - frames extension (default = 'png')

        '''

        # define a video capture object 
        try:

            cap = cv2.VideoCapture(video_path) 
            
            i = 0
            while(True):

                # Capture the video frame 
                # by frame 
                ret, frame = cap.read()
                if ret == False:
                    break
                i +=1
                # images saved in png format
                cv2.imwrite(os.path.join(save_folder, 'img'+str(i)+'.png'),frame)

            
            cap.release()
            cv2.destroyAllWindows()
        
        except Exception as e:
            print('Exception is ', e)
            pass

    