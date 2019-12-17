import numpy as np
from matplotlib.image import imread
from glob import glob
from PIL import Image

from utils.other_utils import RescaleData, ReadTensor

class LoadData:
    def __init__(self, IMG_SHAPE, PATH_DATA, PATH_MASK='inputs/mask/testing_mask_dataset/'):
        self.path_mask = PATH_MASK
        self.path_data = PATH_DATA
        self.dataset, self.data_shape = self.LoadTrainingData(resize=IMG_SHAPE)
        

    def LoadTrainingData(self, resize):
        print('Loading dataset: %s' %self.path_data)
        images = np.array(glob(self.path_data+'*.*'))
        
        if(images[0].endswith('.bin')):
            nr_imgs, imgs_shape = images.size, resize[:-1]
            dataset = np.zeros(tuple(np.append(nr_imgs, np.array(imgs_shape))))
            for i, img in enumerate(images):
                dataset[i] = ReadTensor(filename=img, dimensions=2)
        else:
            nr_imgs, imgs_shape = images.size, imread(images[0]).shape
            if(tuple(resize[:-1]) != imgs_shape):
                print(' --- Inpaint WARNING !!! ---\nprovided shape and images shape does not correspond:\t%s != %s.\nImage are resized to shape provaided in INIT file.' %(str(tuple(resize[:-1])), str(imgs_shape)))
                imgs_shape = resize[:-1]
            dataset = np.zeros(tuple(np.append(nr_imgs, np.array(imgs_shape))))

            for i, img in enumerate(images):
                dataset[i] = np.array(Image.open(img).resize(imgs_shape, Image.ANTIALIAS))

        dataset = dataset[:, :, :, np.newaxis]
        im_shape = dataset.shape[1:-1]      # eg: for mnist (28, 28)
        return dataset, im_shape


    def BatchSample(self, sample, nr_subsample):
        subsample = sample[np.random.randint(0, sample.shape[0], size=nr_subsample)]
        return subsample


    def LoadMaskedData(self, batch):
        maskset = np.zeros(np.append(batch, self.data_shape))
        masked = np.zeros(np.append(batch, self.data_shape))
        batch_data = self.BatchSample(sample=self.dataset, nr_subsample=batch)
        batch_mask = self.BatchSample(sample=np.array(glob(self.path_mask+'*.*')), nr_subsample=batch)      

        # Rescale mask shape to match the images
        for i in range(batch_mask.size):
            mask = Image.open(batch_mask[i])
            mask_resized = np.array(mask.resize(tuple(self.data_shape))) 
            maskset[i] = mask_resized
            masked[i] = np.where(mask_resized == np.max(mask_resized), batch_data.min(), batch_data[i,:,:,0]) # one channel

        # Rescale images values between -1 and 1 
        batch_data = RescaleData(batch_data, a=-1, b=1)
        masked = RescaleData(masked, a=-1, b=1)
        maskset = RescaleData(maskset, a=-1, b=1)

        maskset = maskset[:, :, :, np.newaxis]
        masked = masked[:, :, :, np.newaxis]
        
        return np.array(batch_data), np.array(masked), np.array(maskset)
