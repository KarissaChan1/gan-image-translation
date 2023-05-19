from tensorflow.keras.utils import Sequence
from utils import normalize_vol, augmentation, scale_images
import os 
import numpy as np
import tensorflow as tf

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 path1,
                 path2,
                 batch_size=1,
                 n_channels=1,
                 dim=(256, 256),
                 patch=(32,32),
                 model='pix2pix',
                 rescale=None,
                 shuffle=True,
                 rotation_range=None,
                 shear_range=None,
                 scale_range=None,
                 height_shift_range=None,
                 width_shift_range=None,
                 horizontal_flip=False,
                 vertical_flip=False):
     
        #'Initialization'
        self.dim = dim
        self.patch=patch
        self.batch_size = batch_size
        self.path1 = path1
        self.path2 = path2
        self.model=model
        self.list_IDs1 = os.listdir(self.path1)
        self.list_IDs2 = os.listdir(self.path2)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.rescale = rescale
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.scale_range = scale_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.on_epoch_end()
        
     # Data Augmentation
        self.aug_params = dict()
        self.aug_params['rotation_range'] = self.rotation_range
        self.aug_params['shear_range'] = self.shear_range
        self.aug_params['scale_range'] = self.scale_range
        self.aug_params['height_shift_range'] = self.height_shift_range
        self.aug_params['width_shift_range'] = self.width_shift_range
        self.aug_params['horizontal_flip'] = self.horizontal_flip
        self.aug_params['vertical_flip'] = self.vertical_flip

        if(all(x is None for x in list(self.aug_params.values())[:5]) and
           all(x is False for x in list(self.aug_params.values())[-2:])):
            self.data_aug = False
        else:
            self.data_aug = True

        # Outputs
        print('There are {0} images in the folder.'.format(len(self.list_IDs1)))
        print('There are {0} images in the folder.'.format(len(self.list_IDs2)))

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        if len(self.list_IDs1)>len(self.list_IDs2):
          return int(np.floor(len(self.list_IDs2) / self.batch_size))
        else:
          return int(np.floor(len(self.list_IDs1) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp1 = [self.list_IDs1[k] for k in indexes]
        list_IDs_temp2 = [self.list_IDs2[k] for k in indexes]

        # Generate data
        X,Y = self.__data_generation(list_IDs_temp1,list_IDs_temp2)

        return X,Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if len(self.list_IDs1)>len(self.list_IDs2):
          self.indexes = np.arange(len(self.list_IDs2))
        else:
          self.indexes = np.arange(len(self.list_IDs1))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp1, list_IDs_temp2):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp1):
            # Store sample
            img1 = np.load(self.path1 + ID)
            
            # Normalize data to [-1, 1]
            img1 = normalize_vol(img1,-1,1)

            # Augment data
            if(self.data_aug):
                img1 = augmentation(img1, self.aug_params, self.dim)
             
            # Fill batches
            img1 = tf.reshape(img1, (*self.dim, 1))
            X[i, ] = img1  

        for i, ID in enumerate(list_IDs_temp2):
            # Store sample
            img2 = np.load(self.path2 + ID)
            
            # Normalize data to [-1, 1]
            img2 = normalize_vol(img2,-1,1)

            # Augment data
            if(self.data_aug):
                img2 = augmentation(img2, self.aug_params, self.dim)
             
            # Fill batches
            img2 = tf.reshape(img2, (*self.dim, 1))
            Y[i, ] = img2  

        if self.model=='pix2pix':
            return [X,Y],Y
        else:
            return X,Y
        
        
class RealDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 path,
                 n_channels=1,
                 dim=(256,256),
                 batch_size=256,
                 new_shape=(224,224,3),
                 model='inception',
                shuffle = False):
     
        #'Initialization'
        self.path = path
        self.n_channels=n_channels
        self.dim = dim
        self.list_IDs = os.listdir(self.path)
        self.batch_size = batch_size
        self.new_shape = new_shape
        self.shuffle=shuffle
        self.model = model
        self.on_epoch_end()
        
        print('There are {0} images in the folder.'.format(len(self.list_IDs)))
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X= self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        if self.model=='inception':
            X = np.empty((self.batch_size, *self.new_shape))

            for i, ID in enumerate(list_IDs_temp):
                img = np.load(self.path + ID)
                img = img.reshape((256,256,1))

                # Normalize data to [-1, 1]
                img = normalize_vol(img,-1,1)

                # Scale for InceptionV3
                scaled_img = scale_images(img,self.new_shape)

                # Fill batches
                X[i, ] = scaled_img 
        else:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))

            for i, ID in enumerate(list_IDs_temp):
                img = np.load(self.path + ID)

                # Normalize data to [-1, 1]
                img = normalize_vol(img,-1,1)
             
                # Fill batches
                img1 = tf.reshape(img, (-1, *self.dim, 1))

                # Fill batches
                X[i, ] = img1
#             print(X.shape)
            
        return X