class VideoDataGenerator:

'''
   custom data generator for video augmentation

'''
    def __init__( self,
                  rotation_range=0,
                  width_shift_range=0.0,
                  height_shift_range=0.0,
                  zoom_range=0.0,
                  channel_shift_range=0.0,
                  fill_mode="nearest",
                  horizontal_flip=False,
                  vertical_flip=False,
                  rescale=None,
                  validation_split=0.0,
                  test_split=0.0,
                  dtype=None,

                  )
     '''
     Constructor function for calling video data generator 