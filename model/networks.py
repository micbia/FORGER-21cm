import numpy as np

from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization, Input, Activation, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, AveragePooling2D
from keras.utils import multi_gpu_model, plot_model
from keras import models, optimizers, initializers, callbacks, regularizers

from config.net_config import NetworkConfig
from utils.other_utils import GenerateNoise, GenerateLabels, RescaleData


class NetworkComponents:
    def __init__(self, CONFIG_FILE, PATH_OUTPUT):
        
        # Configure networks
        self.conf = NetworkConfig(CONFIG_FILE)
        self.config_file = CONFIG_FILE
        self.path_output = PATH_OUTPUT


    def Discriminator(self):
        print('Create Adversary network...')

        kinit = initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        img_input = Input(shape=self.conf.img_shape)

        # first downsample
        l1 = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters,
                     strides=2, kernel_initializer=kinit, padding='same')(img_input)
        l1 = LeakyReLU(alpha=0.01)(l1)
        l1 = BatchNormalization(momentum=0.9)(l1)
        l1 = Dropout(self.conf.dropout)(l1)

        # second downsample
        l2 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                     strides=2, kernel_initializer=kinit, padding='same')(l1)
        l2 = LeakyReLU(alpha=0.01)(l2)
        l2 = BatchNormalization(momentum=0.9)(l2)
        l2 = Dropout(self.conf.dropout)(l2)

        # third downsample
        l3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, 
                     strides=2, kernel_initializer=kinit, padding='same')(l2)
        l3 = LeakyReLU(alpha=0.01)(l3)
        l3 = BatchNormalization(momentum=0.9)(l3)
        features_output = Dropout(self.conf.dropout)(l3)

        # Classifier Branch
        label_output = Flatten()(features_output)
        label_output = Dense(self.conf.img_shape[2], activation='sigmoid')(label_output)
        
        #D = models.Model(img_input, [validity_output, label_output])
        D = models.Model(img_input, label_output)
    
        return D

    def Autoencoder(self):
        print('Create Generator-Autoencoder network...')
        A = models.Sequential()

        kinit = initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        # Encoder - first layers
        A.add(Conv2D(filters=int(self.conf.coarse_dim/8), input_shape=self.conf.img_shape,
                     kernel_size=self.conf.filters, kernel_initializer=kinit, padding='same', use_bias=False))
        A.add(BatchNormalization(momentum=0.9))
        A.add(LeakyReLU(alpha=0.01))
        A.add(AveragePooling2D(pool_size=(2, 2)))
        A.add(Dropout(self.conf.dropout))

        # Encoder - second layers
        A.add(Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters,
                     kernel_initializer=kinit, padding='same', use_bias=False))
        A.add(BatchNormalization(momentum=0.9))
        A.add(LeakyReLU(alpha=0.01))
        A.add(AveragePooling2D(pool_size=(2, 2)))
        A.add(Dropout(self.conf.dropout))
        
        # Encoder - third layers
        A.add(Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                     kernel_initializer=kinit, padding='same', use_bias=False))
        A.add(BatchNormalization(momentum=0.9))
        A.add(LeakyReLU(alpha=0.01))
        A.add(Dropout(self.conf.dropout))
        
        # Decoder - first layers
        A.add(Conv2DTranspose(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                              strides=1, kernel_initializer=kinit, padding='same', use_bias=False))
        A.add(LeakyReLU(alpha=0.01))
        A.add(Dropout(self.conf.dropout))

        # Decoder - second layers
        A.add(Conv2DTranspose(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, 
                              strides=2, kernel_initializer=kinit, padding='same', use_bias=False))
        A.add(LeakyReLU(alpha=0.01))
        A.add(Dropout(self.conf.dropout))
        
        # Decoder - tird layers
        A.add(Conv2DTranspose(filters=int(self.conf.coarse_dim/8), kernel_size=self.conf.filters, 
                              strides=2, kernel_initializer=kinit, padding='same', use_bias=False))
        A.add(LeakyReLU(alpha=0.01))
        A.add(Dropout(self.conf.dropout))

        # outro layer
        A.add(Conv2D(filters=int(self.conf.img_shape[2]), kernel_size=self.conf.filters, 
                     kernel_initializer=kinit, padding='same', use_bias=False))
        A.add(Activation('tanh'))
        return A


    def Unet(self):
        print('Create U-Net Generator network...')

        kinit = initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        img_input = Input(shape=self.conf.img_shape, name='Masked_Image')

        # U-Net Encoder - upper level
        e1 = Conv2D(filters=int(self.conf.coarse_dim/8), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='E1_C1')(img_input)
        e1 = BatchNormalization(momentum=0.9, name='E1_BN1')(e1)
        e1 = LeakyReLU(alpha=0.01, name='E1_A1')(e1)
        #e1 = Dropout(self.conf.dropout, name='E1_D1')(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim/8), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='E1_C2')(img_input)
        e1 = BatchNormalization(momentum=0.9, name='E1_BN2')(e1)
        e1c = LeakyReLU(alpha=0.01, name='E1_A2')(e1)
        e1 = Dropout(self.conf.dropout, name='E1_D2')(e1c)
        e1 = AveragePooling2D(pool_size=(2, 2), name='E1_P')(e1c)

        # U-Net Encoder - second level
        e2 = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='E2_C1')(e1)
        e2 = BatchNormalization(momentum=0.9, name='E2_BN1')(e2)
        e2 = LeakyReLU(alpha=0.01, name='E2_A1')(e2)
        #e2 = Dropout(self.conf.dropout, name='E2_D1')(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='E2_C2')(e2)
        e2 = BatchNormalization(momentum=0.9, name='E2_BN2')(e2)
        e2c = LeakyReLU(alpha=0.01, name='E2_A2')(e2)
        e2 = Dropout(self.conf.dropout, name='E2_D2')(e2c)
        e2 = AveragePooling2D(pool_size=(2, 2), name='E2_P')(e2c)
        
        # U-Net Encoder - tird level
        e3 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='E3_C1')(e2)
        e3 = BatchNormalization(momentum=0.9, name='E3_BN1')(e3)
        e3 = LeakyReLU(alpha=0.01, name='E3_A1')(e3)
        #e3 = Dropout(self.conf.dropout, name='E3_D1')(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='E3_C2')(e3)
        e3 = BatchNormalization(momentum=0.9, name='E3_BN2')(e3)
        e3c = LeakyReLU(alpha=0.01, name='E3_A2')(e3)
        e3 = Dropout(self.conf.dropout, name='E3_D2')(e3c)
        e3 = AveragePooling2D(pool_size=(2, 2), name='E3_P')(e3c)
                
        # U-Net Encoder - fourth level
        e4 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='E4_C1')(e3)
        e4 = BatchNormalization(momentum=0.9, name='E4_BN1')(e4)
        e4 = LeakyReLU(alpha=0.01, name='E4_A1')(e4)
        #e4 = Dropout(self.conf.dropout, name='E4_D1')(e4)
        e4 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='E4_C2')(e4)
        e4 = BatchNormalization(momentum=0.9, name='E4_BN2')(e4)
        e4c = LeakyReLU(alpha=0.01, name='E4_A2')(e4)
        e4 = Dropout(self.conf.dropout, name='E4_D2')(e4c)
        '''
        e4 = AveragePooling2D(pool_size=(2, 2), name='E4_P')(e4c)
        
        # VERY IMPORTANT!!!
        # U-Net works best when images are downsampled to the first 1-digit sizes, therefore add or
        # remove a level in order to respect this rule, eg: (128, 128) -> (8, 8) : 2^x=8/128 => x = 4
        # To remember when using 2D-Pk, input shape ~ (250, 250)

        # U-Net - Bottom level
        #b = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
        #            kernel_initializer=kinit, padding='same')(b)
        #b = BatchNormalization(momentum=0.9)(b)
        #b = LeakyReLU(alpha=0.01)(b)
        
        # U-Net Decoder - fourth level
        d4 = Conv2DTranspose(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, 
                             kernel_initializer=kinit, strides=(2, 2), padding='same', use_bias=False, name='D4_DC')(e4)
        d4 = concatenate([d4, e4c], name='merge_layer_e4_A2')
        d4 = Dropout(self.conf.dropout, name='D4_D1')(d4)
        '''
        d4 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='D4_C1')(e4)
        d4 = BatchNormalization(momentum=0.9, name='D4_BN1')(d4)
        d4 = LeakyReLU(alpha=0.01, name='D4_A1')(d4)
        d4 = Dropout(self.conf.dropout, name='D4_D2')(d4)
        d4 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='D4_C2')(d4)
        d4 = BatchNormalization(momentum=0.9, name='D4_BN2')(d4)
        d4 = LeakyReLU(alpha=0.01)(d4)
        d4 = Dropout(self.conf.dropout, name='D4_D3')(d4)

        # U-Net Decoder - third level
        d3 = Conv2DTranspose(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                            kernel_initializer=kinit, strides=(2, 2), padding='same', use_bias=False, name='D3_DC')(d4)
        d3 = concatenate([d3, e3c], name='merge_layer_E3_A2')
        d3 = Dropout(self.conf.dropout, name='D3_D1')(d3)
        d3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='D3_C1')(d3)
        d3 = BatchNormalization(momentum=0.9)(d3)
        d3 = LeakyReLU(alpha=0.01, name='D3_A1')(d3)
        d3 = Dropout(self.conf.dropout, name='D3_D2')(d3)
        d3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='D3_C2')(d3)
        d3 = BatchNormalization(momentum=0.9)(d3)
        d3 = LeakyReLU(alpha=0.01, name='D3_A2')(d3)
        d3 = Dropout(self.conf.dropout, name='D3_D3')(d3)

        # U-Net Decoder - second level
        d2 = Conv2DTranspose(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, 
                            kernel_initializer=kinit, strides=(2, 2), padding='same', use_bias=False, name='D2_DC')(d3)
        d2 = concatenate([d2, e2c], name='merge_layer_e2_A2')
        d2 = Dropout(self.conf.dropout, name='D2_D1')(d2)
        d2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='D2_C1')(d2)
        d2 = BatchNormalization(momentum=0.9, name='D2_BN1')(d2)
        d2 = LeakyReLU(alpha=0.01, name='D2_A1')(d2)
        d2 = Dropout(self.conf.dropout, name='D2_D2')(d2)
        d2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='D2_C2')(d2)
        d2 = BatchNormalization(momentum=0.9, name='D2_BN2')(d2)
        d2 = LeakyReLU(alpha=0.01, name='D2_A2')(d2)
        d2 = Dropout(self.conf.dropout, name='D2_D3')(d2)

        # U-Net Decoder - upper level
        d1 = Conv2DTranspose(filters=int(self.conf.coarse_dim/8), kernel_size=self.conf.filters, 
                            kernel_initializer=kinit, strides=(2, 2), padding='same', use_bias=False, name='D1_DC')(d2)
        d1 = concatenate([d1, e1c], name='merge_layer_e1_A2')
        d1 = Dropout(self.conf.dropout, name='D1_D1')(d1)
        d1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='D1_C1')(d1)
        d1 = BatchNormalization(momentum=0.9, name='D1_BN1')(d1)
        d1 = LeakyReLU(alpha=0.01, name='D1_A1')(d1)
        d1 = Dropout(self.conf.dropout, name='D1_D2')(d1)
        d1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, use_bias=False,
                    kernel_initializer=kinit, padding='same', name='D1_C2')(d1)
        d1 = BatchNormalization(momentum=0.9, name='D1_BN2')(d1)
        d1 = LeakyReLU(alpha=0.01, name='D1_A2')(d1)
        d1 = Dropout(self.conf.dropout, name='D1_D3')(d1)
        # Outro Layer
        output_image = Conv2D(filters=int(self.conf.img_shape[2]), kernel_size=self.conf.filters, 
                              kernel_initializer=kinit, padding='same', use_bias=False, activation='tanh', name='out_C')(d1)
        
        return models.Model(img_input, output_image)