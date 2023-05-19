import os
import numpy as np
from PIL import Image
from keras.utils import Sequence

def load_data(nr_of_channels=1, nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None, subfolder='', generator=True, batch_size=1, path_to_data="/content/drive/MyDrive/UNIT Test data/"):
    trainA_path = os.path.join(path_to_data, subfolder, 'trainA')
    print(trainA_path)
    trainB_path = os.path.join(path_to_data, subfolder, 'trainB')
    testA_path = os.path.join(path_to_data, subfolder, 'testA')
    testB_path = os.path.join(path_to_data, subfolder, 'testB')

    trainA_image_names = os.listdir(trainA_path)
    if nr_A_train_imgs != None:
        trainA_image_names = trainA_image_names[:nr_A_train_imgs]

    trainB_image_names = os.listdir(trainB_path)
    if nr_B_train_imgs != None:
        trainB_image_names = trainB_image_names[:nr_B_train_imgs]

    testA_image_names = os.listdir(testA_path)
    if nr_A_test_imgs != None:
        testA_image_names = testA_image_names[:nr_A_test_imgs]

    testB_image_names = os.listdir(testB_path)
    if nr_B_test_imgs != None:
        testB_image_names = testB_image_names[:nr_B_test_imgs]

    if generator:
        print('Data loading using generator')
        trainA_images = RealDataGenerator(trainA_path,batch_size=batch_size,n_channels=nr_of_channels,test=False)
        trainB_images = RealDataGenerator(trainB_path,batch_size=batch_size,n_channels=nr_of_channels,test=False)
        testA_images = RealDataGenerator(testA_path,batch_size=batch_size,n_channels=nr_of_channels,test=True)
        testB_images = RealDataGenerator(testB_path,batch_size=batch_size,n_channels=nr_of_channels,test=True)
        # return data_sequence(trainA_path, trainB_path, trainA_image_names, trainB_image_names, batch_size=batch_size, nr_of_channels=nr_of_channels)
        return trainA_images, trainB_images, testA_images, testB_images
    else:
        trainA_images = create_image_array(trainA_image_names, trainA_path, nr_of_channels)
        trainB_images = create_image_array(trainB_image_names, trainB_path, nr_of_channels)
        testA_images = create_image_array(testA_image_names, testA_path, nr_of_channels)
        testB_images = create_image_array(testB_image_names, testB_path, nr_of_channels)

        print('Data has been loaded from {} {}'.format(path_to_data, subfolder))
        return {"trainA_images": trainA_images, "trainB_images": trainB_images,
                "testA_images": testA_images, "testB_images": testB_images,
                "trainA_image_names": trainA_image_names,
                "trainB_image_names": trainB_image_names,
                "testA_image_names": testA_image_names,
                "testB_image_names": testB_image_names}

def create_image_array(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'y':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image
                #image = np.array(Image.open(os.path.join(image_path, image_name)))
                image = np.load(os.path.join(image_path, image_name))
                # print('loaded image shape: {}'.format(np.shape(image)))
                image = image[:, :, np.newaxis]
                print('image shape: {}'.format(np.shape(image)))
            else:                   # RGB image
                image = np.array(Image.open(os.path.join(image_path, image_name)))
            image = normalize_array_max(image)
            image_array.append(image)
    print('create array shape: {}'.format(np.shape(image_array)))

    return np.array(image_array)

def normalize_array(array):
    max_value = max(array.flatten())
    array = array / 255
    array = (array - 0.5)*2

    return array

def normalize_array_max(array):
    max_value = max(array.flatten())
    array = array / max_value
    array = (array - 0.5)*2
    return array

class data_sequence(Sequence):

    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size=1, nr_of_channels=1):
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        self.nr_of_channels=nr_of_channels
        self.n = 0
        self.max = self.__len__()
        
        for image_name in image_list_A:
            if image_name[-1].lower() == 'y':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
            else:
                print('TrainA data is not in .npy format')
        for image_name in image_list_B:
            if image_name[-1].lower() == 'y':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))
            else:
                print('TrainB data is not in .npy format')
        # Outputs
        print('There are {0} images in the folder.'.format(len(self.train_A)))
        print('There are {0} images in the folder.'.format(len(self.train_B)))

    def __len__(self):
        # print(int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size)))
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):
        
        # print(self.train_A)
        # print(len(self.train_A))
        # print(len(self.train_B))
        print(idx)
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        
        real_images_A = create_image_array(batch_A, '', self.nr_of_channels)
        print('Shape of batch A: {}'.format(np.shape(real_images_A)))
        real_images_B = create_image_array(batch_B, '', self.nr_of_channels)
        print('Shape of batch B: {}'.format(np.shape(real_images_B)))

        return real_images_A, real_images_B  # input_data, target_data

    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

class RealDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 path,
                 batch_size=256,
                 n_channels=1,
                 test=False):
     
        #'Initialization'
        self.path = path
        self.list_IDs = os.listdir(self.path)
        self.batch_size = batch_size
        self.shuffle=False
        self.dim = (256,256)
        self.test=False
        self.n_channels=n_channels
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
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, ID in enumerate(list_IDs_temp):
            img = np.load(os.path.join(self.path,ID))
            img = img.reshape((256,256,1))
            # else:
            #     if len(np.shape(X)==3):
            #         X = np.squeeze(X,axis=2)

            # Normalize data to [-1, 1]
            img = normalize_array_max(img)
             
            # Fill batches
            X[i, ] = img 
        
        return X

if __name__ == '__main__':
    load_data()
