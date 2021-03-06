import numpy as np, matplotlib.pyplot as plt, os 
import seaborn as sns
from sys import argv
from PIL import Image

from utils.other_utils import GenerateNoise, RescaleData, ReadTensor


class Plots:
    def __init__(self, IMG_SHAPE, PATH_DATA, PATH_MASK, PATH_OUTPUT):
        self.path_data = PATH_DATA
        self.path_mask = PATH_MASK
        self.path_output = PATH_OUTPUT
        self.img_shape = IMG_SHAPE

        self.mask_arr = ['00249.png', '00261.png', '03410.png', '00296.png', '00284.png', '00344.png', '00477.png', '00472.png', '00510.png', '00641.png']

    def PlotLosses(self, epch, prev_epch, loss_D_real, loss_D_fake, loss_G, lr):
        # Plot losses 
        fig, ax1 = plt.subplots(figsize=(10, 8))
        ax1.plot(loss_D_real, c='tab:blue', label='Adversary loss - Real')
        ax1.plot(loss_D_fake, c='tab:orange' ,label='Adversary loss - Fake')
        ax1.plot(loss_G, c='tab:green', label='Generator loss')
        
        ax2 = ax1.twinx()
        ax2.semilogy(lr, c='gray', ls='--', alpha=0.5)
        ax2.set_ylabel('Learning Rate')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        if(prev_epch != 0):
            os.system('rm %simages/loss_ep-*.png' %(self.path_output))
        plt.savefig('%simages/loss_ep-%d.png' %(self.path_output, epch+1), bbox_inches='tight')
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
        plt.savefig('%simages/generated_test/gan_generated_image_ep-%d.png' %(self.path_output, epch+1), bbox_inches='tight')
        plt.close('all')

    def PlotDistribution(self, epch, label_fake, data_fake, label_real, data_real, label_reconst, data_reconst, nl=5):
        # Plot generated images
        plt.figure(figsize=(12, 12))
        sns.kdeplot(data_fake, label_fake, n_levels=nl, color='blue', label='Adversary - PDF fake')
        sns.kdeplot(data_fake, label_fake, n_levels=nl, cmap='Blues', shade=True, shade_lowest=False)

        sns.kdeplot(data_real, label_real, n_levels=nl, color='red', label='Adversary - PDF real')
        sns.kdeplot(data_real, label_real, n_levels=nl, cmap='Oranges', shade=True, shade_lowest=False)
        
        plt.scatter(x=data_reconst, y=label_reconst, marker='x', color='black', label='GAN - PDF output')
        plt.title('Epoch:%3d' %(epch+1), size=20), plt.xlabel('Predicted label', size=16), plt.ylabel('True label', size=16), plt.legend(loc='lower right')
        plt.xlim(-0.1, 1.3), plt.ylim(-0.1, 1.3)
        plt.savefig('%simages/pdf_gan/pdf_ep-%d.png' %(self.path_output, epch+1), bbox_inches='tight')
        plt.close('all')

    def PlotSpecificMNISTImages(self, epch, gmodel, dim=(10, 10)):
        # Plot generated images, for mnist specificly
        img_arr = ['10010.png', '40860.png', '04476.png', '40858.png', '40842.png', '40869.png', '40862.png', '40873.png', '40852.png', '40854.png']

        fig, ax = plt.subplots(len(img_arr), 3, figsize=(8, 18))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.gray()
        
        for i in range(len(img_arr)):
            img = np.array(Image.open(self.path_data+img_arr[i]).resize(self.img_shape[:-1], Image.ANTIALIAS))
            mask = np.array(Image.open(self.path_mask+self.mask_arr[i]).resize(img.shape))
            masked = np.where(mask>np.mean(mask)*0.7, img.max(), img)
            
            masked = RescaleData(masked[np.newaxis, :, :, np.newaxis], a=-1, b=1)
            reconstr = gmodel.predict(masked)

            ax[i,0].imshow(masked[0,:,:,0])
            ax[i,0].get_xaxis().set_visible(False)
            ax[i,0].get_yaxis().set_visible(False)
                
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


    def PlotSpecificImages(self, epch, gmodel, dim=(10, 10)):
            # Plot generated images, similar to mnist plotting
            img_arr = ['image_xi0j149_500Mpc.bin', 'image_xi32j149_500Mpc.bin', 'image_xi0j249_244Mpc.bin', 'image_xi40j252_244Mpc.bin', 'image_zi0j128_64Mpc.bin', 'image_xi0j127_64Mpc.bin']
            #img_arr = ['image_xi0j149_m500Mpc.bin', 'image_xi32j149_m500Mpc.bin', 'image_xi0j249_244Mpc.bin', 'image_xi40j252_244Mpc.bin', 'image_zi0j128_256_64Mpc.bin', 'image_xi0j254_512_64Mpc.bin']

            fig, ax = plt.subplots(len(img_arr), 3, figsize=(8, 18))
            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            plt.jet()
            
            #mask = 

            for i in range(len(img_arr)):
                img = ReadTensor(filename=self.path_data+img_arr[i], dimensions=2)
                mask = np.array(Image.open(self.path_mask+self.mask_arr[i]).resize(img.shape))
                masked = np.where(mask == np.max(mask), img.min(), img)
                
                img = RescaleData(img[np.newaxis, :, :, np.newaxis], a=-1, b=1)
                mask = RescaleData(mask[np.newaxis, :, :, np.newaxis], a=-1, b=1)
                masked = RescaleData(masked[np.newaxis, :, :, np.newaxis], a=-1, b=1)
                reconstr = gmodel.predict([img, mask])

                ax[i,0].imshow(masked[0,:,:,0])
                ax[i,0].get_xaxis().set_visible(False)
                ax[i,0].get_yaxis().set_visible(False)
         
                ax[i,1].imshow(reconstr[0,:,:,0])
                ax[i,1].get_xaxis().set_visible(False)
                ax[i,1].get_yaxis().set_visible(False)

                ax[i,2].imshow(img[0,:,:,0])
                ax[i,2].get_xaxis().set_visible(False)
                ax[i,2].get_yaxis().set_visible(False)

            fig.suptitle('Epoch: %4d' %(epch+1), fontsize=25)
            ax[0,0].set_title('masked', size=17)
            ax[0,1].set_title('reconstructed', size=17)
            ax[0,2].set_title('original', size=17)
            fig.tight_layout()
            fig.subplots_adjust(top=0.94)
            plt.savefig('%simages/generated_test/test_reconst_epoch_%d.png' %(self.path_output, epch+1), bbox_inches='tight')
            plt.close('all')