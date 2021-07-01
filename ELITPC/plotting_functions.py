import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

#####################################################
#####################################################
def get_ROI_bb(roi):
    sx, sy = roi['slice'] #Use transosed coordinates for plotting
    #print("ROI: ({},{}) - ({},{})".format(sx.start,sy.start, sx.stop,sy.stop))
        
    scalex = 1.0/512
    scaley = 1.0/92
        
    width = (sx.stop-sx.start)*scalex
    height = (sy.stop-sy.start)*scaley
    (startx,starty) = (sx.start*scalex,sy.start*scaley)
    
    return Rectangle((startx,starty), width, height,
                    linewidth=1, edgecolor='red',facecolor='none') 
#####################################################
#####################################################
def plotOriginal_vs_cropped(data, cropped, mask, roi):
    
    if roi == None:
        return
    bb = get_ROI_bb(roi)
    
    projection = 0
    
    fig, ax = plt.subplots(1,3, figsize=(20,8))
    #imageRange = [0,511, 0,91]  
    imageRange = [0,1, 0,1]  
    im0 = ax[0].imshow(np.transpose(data[:,:,projection]) ,extent=imageRange, origin='lower')
    ax[0].add_patch(bb)
    im1 = ax[1].imshow(np.transpose(cropped[:,:,projection]) ,extent=imageRange, origin='lower') 
    im1 = ax[2].imshow(np.transpose(mask[:,:,projection]) ,extent=imageRange, origin='lower') 
    
    ax[0].set_title('original', fontsize = 15)
    ax[1].set_title('cropped', fontsize = 15) 
    ax[2].set_title('mask', fontsize = 15) 
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.4)
    fig.colorbar(im0, cax=cax)
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.4)
    fig.colorbar(im1, cax=cax)
    
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.4)
    fig.colorbar(im1, cax=cax)
#####################################################
#####################################################
def plotOriginal_vs_decoded(data, vae):
    
    batchElement = 0
    projection = 0
    
    z_mean, z_log_var, z = vae.encoder.predict(data)
    decoded_data = vae.decoder.predict(z_mean)  
    decoded_data = tf.cast(decoded_data[batchElement],tf.float32)
    data = tf.cast(data[batchElement],tf.float32)
    
    fig, ax = plt.subplots(1,2, figsize=(20,8))
    #imageRange = [0,511, 0,91]  
    imageRange = [0,1, 0,1]  
    im0 = ax[0].imshow(np.transpose(data[:,:,projection]) ,extent=imageRange, origin='lower')
    im1 = ax[1].imshow(np.transpose(decoded_data[:,:,projection]) ,extent=imageRange, origin='lower') 
    
    ax[0].set_title('original', fontsize = 15)
    ax[1].set_title('decoded, z = {}'.format(z_mean), fontsize = 15)  
    
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.4)
    fig.colorbar(im0, cax=cax)
    
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.4)
    fig.colorbar(im1, cax=cax)
#####################################################
#####################################################  
def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 64
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
#####################################################
#####################################################
def plotLoss(data):
    
    fig, axes = plt.subplots(1, 4, figsize = (15, 5))

    data.hist("binary_loss", bins=20, ax=axes[0])
    axes[0].set_ylabel('Number of events')
    axes[0].set_xlabel('Reconstruction loss')

    data.hist("binary_loss", bins=20, ax=axes[1])
    axes[1].set_ylabel('Number of events')
    axes[1].set_xlabel('Reconstruction loss')
    axes[1].set_yscale("log")
    
    data.hist("mae_loss", bins=20, ax=axes[2])
    axes[2].set_ylabel('Number of events')
    axes[2].set_xlabel('Reconstruction loss')

    data.hist("mae_loss", bins=20, ax=axes[3])
    axes[3].set_ylabel('Number of events')
    axes[3].set_xlabel('Reconstruction loss')
    axes[3].set_yscale("log")
#####################################################
#####################################################   