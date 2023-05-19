from scipy.linalg import sqrtm
import numpy as np
import tensorflow as tf
from scipy.special import rel_entr

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    
    # images1: datagenerator object of real images
    # images2: array of fake generated images
    
    act1 = model.predict_generator(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
    
    
def get_psnr(real, generated):
    psnr_value = tf.reduce_mean(tf.image.psnr(generated, real, max_val=1.0))
    return psnr_value
    
def get_ssim(real, generated):
    ssim = tf.reduce_mean(tf.image.ssim(real, generated, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03))
    return ssim
    
    
def get_hs(real, generated):
    hist_real,_ = np.histogram(real,bins=256,range=(0.01,1),density=True)
    hist_fake,_ = np.histogram(generated,bins=256,range=(0.01,1),density=True)
    kl = sum(rel_entr(hist_fake,hist_real))
    return kl