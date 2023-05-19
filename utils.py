from scipy import ndimage
import numpy as np
from scipy.linalg import sqrtm
from skimage.transform import resize

# Keras Augmentation Functions
def flip_axis(x, axis):

    '''
    Description:
    Function that flips the axes of a given input.

    Arguments:
    - x(npy): input image/volume

    Returns:
    - x(npy): modified image/volume
    '''

    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)

    return x

def transform_matrix_offset_center(matrix, x, y):

    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)

    return transform_matrix

def get_random_transform(seed=None, img_size=(256, 256), rotation_range=None, 
                         height_shift_range=None, width_shift_range=None,
                         shear_range=None, scale_range=None,
                         horizontal_flip=False, vertical_flip=False):

    # Random number seed
    if(seed is not None):
        np.random.seed(seed)

    # Check rotation range
    if(rotation_range):
        theta = np.random.uniform(*rotation_range)
    else:
        theta = 0

    # Vertical translation
    if(height_shift_range):
        if(type(height_shift_range) == tuple):
            tx = np.random.uniform(*height_shift_range)
        else:
            height_shift_range = int(height_shift_range * img_size[0])
            tx = np.random.uniform(-height_shift_range, height_shift_range)
    else:
        tx = 0

    # Horizontal translation
    if(width_shift_range):
        if(type(width_shift_range) == tuple):
            ty = np.random.uniform(*width_shift_range)
        else:
            width_shift_range = int(width_shift_range * img_size[1])
            ty = np.random.uniform(-width_shift_range, width_shift_range)
    else:
        ty = 0

    # Shearing range
    if(shear_range):
        shear = np.random.uniform(*shear_range)
    else:
        shear = 0

    # Scaling range
    if(scale_range):
        sx, sy = np.random.uniform(*scale_range, 2)
    else:
        sx, sy = 1, 1

    # Horizontal and vertical flip
    flip_h = (np.random.random() < 0.5) * horizontal_flip
    flip_v = (np.random.random() < 0.5) * vertical_flip

    # Dictionary of transformation parameters
    transform_parameters = {'theta': theta,
                            'tx': tx,
                            'ty': ty,
                            'sx': sx,
                            'sy': sy,
                            'shear': shear,
                            'flip_h': flip_h,
                            'flip_v': flip_v}

    return transform_parameters


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, sx=1, sy=1,
                           row_axis=0, col_axis=1, channel_axis=2, order=1,
                           cval=0., fill_mode='nearest'):

    # Rotation
    transform_matrix = None
    if(theta != 0):
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    # Translation
    if(tx != 0 or ty != 0):
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    # Shearing
    if(shear != 0):
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    # Scaling
    if(sx != 1 or sy != 1):
        scale_matrix = np.array([[sx, 0, 0],
                                [0, sy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = scale_matrix
        else:
            transform_matrix = np.dot(transform_matrix, scale_matrix)

    # Apply affine transformation
    if(transform_matrix is not None):
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)

    return x


def apply_transform(x, tform, img_row_axis=0, img_col_axis=1, img_channel_axis=2):

    x = apply_affine_transform(x, tform.get('theta', 0),
                                  tform.get('tx', 0),
                                  tform.get('ty', 0),
                                  tform.get('shear', 0),
                                  tform.get('sx', 1),
                                  tform.get('sy', 1),
                                  row_axis=img_row_axis,
                                  col_axis=img_col_axis,
                                  channel_axis=img_channel_axis)

    if(tform.get('flip_h', False)):
        x = flip_axis(x, img_col_axis)

    if(tform.get('flip_v', False)):
        x = flip_axis(x, img_row_axis)

    return x

def augmentation(img, params, img_size=(256, 256)):

    tform = get_random_transform(**params, img_size=img_size)
    img_aug = apply_transform(img, tform)

    return img_aug


def normalize_vol(vol,min_range=-1,max_range=1):
    # Cast to float
    vol = vol.astype(np.float32)

    # Normalize to [-1, 1]
    I_max = np.max(vol)
    I_min = np.min(vol)

    vol = (max_range - min_range)*((vol - I_min) / (I_max - I_min)) + min_range
    return vol
  
def remove_slices(vol,vol2,vol3,threshold=0.1):
    num_slices = vol.shape[2]
    total_pix = vol.shape[0] * vol.shape[1]
    idx_array1, idx_array2, idx_array3 = [],[],[]

    for k in range(num_slices):
        counter1,counter2,counter3 = 0,0,0

        for i in range(vol.shape[0]):
            for j in range(vol.shape[1]):
                if vol[i,j,k] > 0:
                    counter1+=1
                if vol2[i,j,k] > 0:
                    counter2+=1
                if vol3[i,j,k] > 0:
                    counter3+=1

        if counter1 > total_pix*threshold:
            idx_array1.append(k)
        if counter2 > total_pix*threshold:
            idx_array2.append(k)
        if counter3 > total_pix*threshold:
            idx_array3.append(k)

    idx_arrays = [idx_array1,idx_array2,idx_array3]
    len_arrays = [len(idx_array1),len(idx_array2),len(idx_array3)]

    min_idx = np.argmin(len_arrays)

    new_vol = vol[:,:,idx_arrays[min_idx]]
    new_vol2 = vol2[:,:,idx_arrays[min_idx]]
    new_vol3 = vol3[:,:,idx_arrays[min_idx]]
    
    new_vol = new_vol[:,:,0:new_vol.shape[2]]
    new_vol2 = new_vol2[:,:,0:new_vol2.shape[2]]
    new_vol3 = new_vol3[:,:,0:new_vol3.shape[2]]
    
    
    return new_vol,new_vol2,new_vol3
    
    
    
def save_slices(split,split_dir):
    vol = np.zeros((3,256,256,55))
    image_types = ['FA','MD','FLAIR']
    list_of_files = []

    for i in range(len(split)):
        count=0
        for t in image_types:
            new_folder = split_dir+t+'/'
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)

            im_dict = sio.loadmat(rootpath+t+"/"+split[t][i])
            vol_name = list(im_dict.keys())[-1]
            vol[count,:,:,:] = im_dict[vol_name]
            count+=1

        new_fa,new_md,new_flair = remove_slices(vol[0,:,:,:],vol[1,:,:,:],vol[2,:,:,:],threshold=0.2)

        for s in range(new_fa.shape[2]):
            im_fa = new_fa[:,:,s]
            im_md = new_md[:,:,s]
            im_flair = new_flair[:,:,s]

            fa_file_name = split_dir+'FA/' + split['Patient_ID'][i] + '_slice_{0}'.format(str(s)) 
            md_file_name = split_dir+'MD/'+ split['Patient_ID'][i] + '_slice_{0}'.format(str(s)) 
            flair_file_name = split_dir+'FLAIR/'+ split['Patient_ID'][i] + '_slice_{0}'.format(str(s)) 
            np.save(fa_file_name,im_fa)
            np.save(md_file_name,im_md)
            np.save(flair_file_name,im_flair)

            list_of_files.append(split['Patient_ID'][i] + '_slice_{0}'.format(str(s)))
  
    print("Images saved!")
    return list_of_files
    

def scale_images(image, new_shape):
    # resize with nearest neighbor interpolation
    new_image = resize(image, new_shape, 0)
    return new_image

