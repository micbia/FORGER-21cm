import numpy as np, matplotlib.pyplot as plt, os 
from sys import argv
from PIL import Image

from utils.other_utils import GenerateNoise, RescaleData


class Plots:
    def __init__(self, IMG_SHAPE, PATH_DATA, PATH_MASK, PATH_OUTPUT):
        self.path_data = PATH_DATA
        self.path_mask = PATH_MASK
        self.path_output = PATH_OUTPUT
        self.img_shape = IMG_SHAPE

        self.img_arr = ['10010.png', '40860.png', '40841.png', '40858.png', '40842.png', '40869.png', '40862.png', '40873.png', '40852.png', '40854.png']
        self.mask_arr = ['00249.png', '00261.png', '00261.png', '00296.png', '00284.png', '00344.png', '00477.png', '00472.png', '00510.png', '00641.png']

    def PlotLosses(self, epch, prev_epch, loss_D_real, loss_D_fake, loss_G):
        # Plot losses 
        plt.figure(figsize=(10, 8))
        plt.plot(loss_D_real, c='tab:blue', label='Adversary loss - Real')
        plt.plot(loss_D_fake, c='tab:orange' ,label='Adversary loss - Fake')
        plt.plot(loss_G, c='tab:green', label='Generator loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if(prev_epch != 0):
            os.remove('%simages/gan_loss_ep-*.png' %(self.path_output))
        plt.savefig('%simages/gan_loss_ep-%d.png' %(self.path_output, epch), bbox_inches='tight')
        plt.close('all')

    def PlotGeneratedImages(self, epch, gmodel, input_dim, examples=100, dim=(10, 10)):
        # Plot generated images
        plt.figure(figsize=(12, 12))
        noise = GenerateNoise(examples, input_dim)
        generatedImages = gmodel.predict(noise)
        for i in range(generatedImages.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generatedImages[i, :, :, 0], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.savefig('%simages/generated_test/gan_generated_image_epoch_%d.png' %(self.path_output, epch), bbox_inches='tight')
        plt.close('all')

    def PlotSpecificGeneratedImages(self, epch, gmodel, dim=(10, 10)):
        # Plot generated images, for mnist specificly
        fig, ax = plt.subplots(10,3, figsize=(8, 18))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.gray()
        
        for i in range(10):
            img = np.array(Image.open(self.path_data+self.img_arr[i]).resize(self.img_shape[:-1], Image.ANTIALIAS))
            mask = np.array(Image.open(self.path_mask+self.mask_arr[i]).resize(img.shape))
            masked = np.where(mask>np.mean(mask)*0.7, img.max(), img)
   
            ax[i,0].imshow(masked)
            ax[i,0].get_xaxis().set_visible(False)
            ax[i,0].get_yaxis().set_visible(False)
            
            masked = RescaleData(masked[np.newaxis, :, :, np.newaxis], a=-1, b=1)
            reconstr = gmodel.predict(masked)
         
            ax[i,1].imshow(reconstr[0,:,:,0])
            ax[i,1].get_xaxis().set_visible(False)
            ax[i,1].get_yaxis().set_visible(False)

            ax[i,2].imshow(img)
            ax[i,2].get_xaxis().set_visible(False)
            ax[i,2].get_yaxis().set_visible(False)

        fig.suptitle('Epoch: %4d' %(epch+1), fontsize=25)
        ax[0,0].set_title('masked', size=17)
        ax[0,1].set_title('reconstructed', size=17)
        ax[0,2].set_title('original', size=17)
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        plt.savefig('%simages/generated_test/test_reconst_epoch_%d.png' %(self.path_output, epch), bbox_inches='tight')
        plt.close('all')