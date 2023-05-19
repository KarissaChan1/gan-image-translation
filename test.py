import tensorflow as tf
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import sys
from CycleGAN import define_generator, define_discriminator, CycleGan
from evaluation_metrics import get_mse, get_psnr, get_hs, get_ssim

experiment = 'experiment7_1'
save_root = 'E:/Image Synthesis/experiments/' + experiment

tmpdataDir = 'E:/Image Synthesis/experiments/experiment7_paired/'
test_dir = tmpdataDir + 'test/'

testing_dir = save_root+"/model/testing/"
os.mkdir(testing_dir)

model_chkpt_dir = save_root+"/model/checkpoints/"

input_dir = test_dir+'FLAIR/'
filenames = os.listdir(input_dir)
real_dir = test_dir+'MD/'

f  = open(save_root+'/model/training/training.log', "r")
X = f.read().splitlines()


fidscores = []
for count in range(1,len(X)):
    line = X[count].split(",")
    fidscores.append(line[5])

epoch = fidscores.index(min(fidscores))
optimal_epoch = model_chkpt_dir + 'cyclegan_checkpoints.0{0}'.format(str(epoch+1))

# Load trained model

image_shape = (256,256,1)
gen_G = define_generator(image_shape,name="generator_G")
gen_F = define_generator(image_shape,name="generator_F")
disc_X = define_discriminator(image_shape,name="discriminator_X")
disc_Y = define_discriminator(image_shape,name="discriminator_Y")
trained_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

trained_model.load_weights(optimal_epoch)

# Predict on test set
save_testing='E:/Image Synthesis/experiments/experiment10_1/model/testing/Synthetic Images/'
# os.makedirs(save_testing)

psnr = np.zeros([len(filenames),1])
ssim = np.zeros([len(filenames),1])
hs = np.zeros([len(filenames),1])
mse = np.zeros([len(filenames),1])

# fig1, ax = plt.subplots(10, 4, figsize=(8, 15))
i=0

for f in filenames:
    im_real = np.load(real_dir + f)
    im_real2 = im_real.reshape((256,256,1))
    im_real2 = tf.image.convert_image_dtype(im_real2, tf.float32)
    
    im_input = np.load(input_dir + f)
    img = tf.reshape(im_input, (-1, 256, 256, 1))
    
    prediction = trained_model.gen_G(img)
    prediction = tf.reshape(prediction,(256,256,1))
    prediction = tf.convert_to_tensor(prediction)
    prediction = tf.image.convert_image_dtype(prediction, tf.float32)
    prediction = normalize_vol(prediction.numpy(),min_range = 0, max_range = np.max(im_real2))
    np.save(save_testing+f,prediction)
      
    psnr[i,0] = get_psnr(im_real2,prediction,normalize=True)
    ssim[i,0] = get_ssim(im_real2,prediction,normalize=True)
    hs[i,0],hist_real,hist_fake = get_hs(im_real2,prediction,min_val=0,max_val=np.max(im_real2))
    mse[i,0] = get_mse(im_real2.numpy(),prediction,normalize=True)
    
    # plot sample images
    if i<10:

        ax[i, 0].imshow(im_input,cmap='gray')
        ax[i, 1].imshow(im_real2,cmap='gray')
        ax[i, 2].imshow(prediction,cmap='gray')

        ax[i,3].plot(hist_real[1:])
        ax[i,3].plot(hist_fake[1:])
        
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
        ax[i, 2].axis("off")
        
        if i<1:
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Real image")
            ax[i, 2].set_title("Translated image")
            ax[i, 3].set_title("Histograms")
    
    i+=1
    
plt.savefig(testing_dir+"outputvols_{}.png".format(str(epoch+1)))

print("mean psnr: ", np.mean(psnr))
print("mean ssim: ", np.mean(ssim))
print("mean hist-kl: ", np.mean(hs))
print("mean mse: ", np.mean(mse))

psnr=[element for sublist in psnr for element in sublist]
ssim=[element for sublist in ssim for element in sublist]
hist=[element for sublist in hs for element in sublist]
mse=[element for sublist in mse for element in sublist]

eval_metrics = pd.DataFrame(columns=['psnr','ssim','hist-kl','mse'])
eval_metrics['psnr']=psnr
eval_metrics['ssim']=ssim
eval_metrics['hist-kl']=hist
eval_metrics['mse']=mse
eval_metrics.to_csv(testing_dir+'eval_metrics_{}_2.csv'.format(str(epoch+1)))


