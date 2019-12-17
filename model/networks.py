import numpy as np

from keras import backend as K
from keras.layers import Multiply, Dense, Dropout, Flatten, Reshape, BatchNormalization, UpSampling2D, Input, Activation, ELU, Concatenate, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D
from keras.utils import multi_gpu_model, plot_model
from keras import models, optimizers, initializers, callbacks, regularizers

from config.net_config import NetworkConfig
from utils.other_utils import GenerateNoise, GenerateLabels, RescaleData
from utils.metrics import NegativeLayer

class NetworkComponents:
    def __init__(self, CONFIG_FILE, PATH_OUTPUT):
        
        # Configure networks
        self.conf = NetworkConfig(CONFIG_FILE)
        self.config_file = CONFIG_FILE
        self.path_output = PATH_OUTPUT


    def Discriminator(self):
        print('Create Adversary network ...')

        img_input = Input(shape=self.conf.img_shape)

        # Downsample convolutions
        d = Conv2D(filters=int(self.conf.coarse_dim/8), kernel_size=self.conf.filters,
                   strides=2, kernel_initializer="he_normal", padding='same')(img_input)
        #d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        #d = Dropout(self.conf.dropout)(d)
        d = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, 
                   strides=2, kernel_initializer="he_normal", padding='same')(d)
        #d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        #d = Dropout(self.conf.dropout)(d)
        d = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                   strides=2, kernel_initializer="he_normal", padding='same')(d)
        #d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.01)(d)
        #d = Dropout(self.conf.dropout)(d)
        d = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, 
                   strides=1, kernel_initializer="he_normal", padding='same')(d)
        #d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        #d = Dropout(self.conf.dropout)(d)
        d = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                   strides=1, kernel_initializer="he_normal", padding='same')(d)
        #d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        #d = Dropout(self.conf.dropout)(d)
        d = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, 
                   strides=1, kernel_initializer="he_normal", padding='same')(d)
        #d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        #d = Dropout(self.conf.dropout)(d)

        # Classifier Branch
        d = Flatten()(d)
        label_output = Dense(self.conf.img_shape[2], kernel_initializer="he_normal", activation='sigmoid')(d)
        
        return models.Model(inputs=img_input, outputs=label_output, name='Discriminator')


    def LocalDiscriminator(self):
        print('Create Local Adversary network...')

        img_input = Input(shape=self.conf.img_shape, name='input_img')
        mask_input = Input(shape=self.conf.img_shape, name='input_mask')

        comb_input = Multiply(name='masked_region')([img_input, mask_input])

        # Downsample convolutions
        ld = Conv2D(filters=int(self.conf.coarse_dim/8), kernel_size=self.conf.filters,
                   strides=2, kernel_initializer="he_normal", padding='same')(comb_input)
        ld = LeakyReLU()(ld)
        ld = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, 
                   strides=2, kernel_initializer="he_normal", padding='same')(ld)
        ld = LeakyReLU()(ld)
        ld = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                   strides=2, kernel_initializer="he_normal", padding='same')(ld)
        ld = LeakyReLU(alpha=0.01)(ld)
        ld = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=self.conf.filters, 
                   strides=1, kernel_initializer="he_normal", padding='same')(ld)
        ld = LeakyReLU()(ld)
        ld = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                   strides=1, kernel_initializer="he_normal", padding='same')(ld)
        ld = LeakyReLU()(ld)
        ld = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, 
                   strides=1, kernel_initializer="he_normal", padding='same')(ld)
        ld = LeakyReLU()(ld)

        # Classifier Branch
        ld = Flatten()(ld)
        #ld = Dense(1024, kernel_initializer="he_normal", activation='tanh')(ld)
        label_output = Dense(self.conf.img_shape[2], kernel_initializer="he_normal", activation='sigmoid')(ld)
        
        return models.Model(inputs=[img_input, mask_input], outputs=label_output, name='Local_Discriminator')


    def GlobalDiscriminator(self):
        print('Create Global Adversary network...')

        img_input = Input(shape=self.conf.img_shape, name='entire_picture')

        # first downsample
        gd = Conv2D(filters=int(self.conf.coarse_dim/8), kernel_size=self.conf.filters,
                     strides=2, kernel_initializer="he_normal", padding='same')(img_input)
        gd = LeakyReLU(alpha=0.01)(gd)

        # second downsample
        gd = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, 
                     strides=2, kernel_initializer="he_normal", padding='same')(gd)
        gd = LeakyReLU(alpha=0.01)(gd)

        # third downsample
        gd = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                     strides=2, kernel_initializer="he_normal", padding='same')(gd)
        gd = LeakyReLU(alpha=0.01)(gd)

        # Classifier Branch
        gd = Flatten()(gd)
        #gd = Dense(1024, kernel_initializer="he_normal", activation='tanh')(gd)
        label_output = Dense(self.conf.img_shape[2], kernel_initializer="he_normal", activation='sigmoid')(gd)
        
        return models.Model(inputs=img_input, outputs=label_output, name='Global_Discriminator')


    def Autoencoder(self):
        print('Create Generator-Autoencoder network...')

        inputs_img = Input(shape=self.conf.img_shape)
        input_mask = Input(shape=self.conf.img_shape)
        
        neg_mask = NegativeLayer()(input_mask)
        inputs = Multiply()([inputs_img, neg_mask])
        
        inputs = Concatenate(axis=3)([inputs, input_mask])

        # Encoder - first layers
        a = Conv2D(filters=int(self.conf.coarse_dim/8), input_shape=self.conf.img_shape,
                    kernel_size=self.conf.filters, kernel_initializer="he_normal", padding='same', use_bias=False)(inputs)
        a = BatchNormalization(momentum=0.9)(a)
        a = LeakyReLU(alpha=0.01)(a)
        a = AveragePooling2D(pool_size=(2, 2))(a)
        a = Dropout(self.conf.dropout)(a)

        # Encoder - second layers
        a = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters,
                    kernel_initializer="he_normal", padding='same', use_bias=False)(a)
        a = BatchNormalization(momentum=0.9)(a)
        a = LeakyReLU(alpha=0.01)(a)
        a = AveragePooling2D(pool_size=(2, 2))(a)
        a = Dropout(self.conf.dropout)(a)
        
        # Encoder - third layers
        a = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                    kernel_initializer="he_normal", padding='same', use_bias=False)(a)
        a = BatchNormalization(momentum=0.9)(a)
        a = LeakyReLU(alpha=0.01)(a)
        a = Dropout(self.conf.dropout)(a)
        
        # Decoder - first layers
        a = Conv2DTranspose(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                            strides=1, kernel_initializer="he_normal", padding='same', use_bias=False)(a)
        a = LeakyReLU(alpha=0.01)(a)
        a = Dropout(self.conf.dropout)(a)

        # Decoder - second layers
        a = Conv2DTranspose(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, 
                            strides=2, kernel_initializer="he_normal", padding='same', use_bias=False)(a)
        a = LeakyReLU(alpha=0.01)(a)
        a = Dropout(self.conf.dropout)(a)
        
        # Decoder - tird layers
        a = Conv2DTranspose(filters=int(self.conf.coarse_dim/8), kernel_size=self.conf.filters, 
                            strides=2, kernel_initializer="he_normal", padding='same', use_bias=False)(a)
        a = LeakyReLU(alpha=0.01)(a)
        a = Dropout(self.conf.dropout)(a)

        # outro layer
        a = Conv2D(filters=int(self.conf.img_shape[2]), kernel_size=self.conf.filters, 
                    kernel_initializer="he_normal", padding='same', use_bias=False)(a)
        reconstr_img = Activation('tanh')(a)
        
        return models.Model(inputs=[inputs_img, input_mask], outputs=[reconstr_img], name='Generator_Autoencoder')


    def Auto3Encoder1Decoder(self):
        print('Create Generator-Autoencoder network with 3 Encoder and 1 Decoder...')

        inputs_img = Input(shape=self.conf.img_shape, name='input_img')
        input_mask = Input(shape=self.conf.img_shape, name='input_mask')
        
        neg_mask = NegativeLayer()(input_mask)
        inputs = Multiply()([inputs_img, neg_mask])
        
        inputs = Concatenate(axis=3, name='inputs_encoder')([inputs, input_mask])

        # Econder 1 
        e1 = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(inputs)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=7, strides=(2, 2), 
                    kernel_initializer="he_normal", padding='same')(e1)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e1)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=7, strides=(2, 2), 
                    kernel_initializer="he_normal", padding='same')(e1)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e1)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e1)

        e1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(2, 2), padding='same')(e1)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(4, 4), padding='same')(e1)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(8, 8), padding='same')(e1)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(16, 16), padding='same')(e1)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e1)
        e1 = ELU()(e1)
        e1 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=7, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e1)
        e1 = ELU()(e1)
        e1out = UpSampling2D(size=(4, 4))(e1)


        # Encoder 2
        e2 = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(inputs)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=5, strides=(2, 2), 
                    kernel_initializer="he_normal", padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(2, 2), 
                    kernel_initializer="he_normal", padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e2)

        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(2, 2), padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(4, 4), padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(8, 8), padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(16, 16), padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e2)
        e2 = ELU()(e2)
        
        e2 = UpSampling2D(size=(2, 2))(e2)

        e2 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e2)
        e2 = ELU()(e2)
        e2 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=5, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e2)
        e2 = ELU()(e2)

        e2out = UpSampling2D(size=(2, 2))(e2)


        # Encoder 3
        e3 = Conv2D(filters=int(self.conf.coarse_dim/4), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(inputs)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=3, strides=(2, 2), 
                    kernel_initializer="he_normal", padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=3, strides=(2, 2), 
                    kernel_initializer="he_normal", padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e3)

        e3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(2, 2), padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(4, 4), padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(8, 8), padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", dilation_rate=(16, 16), padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e3)
        e3 = ELU()(e3)
        
        e3 = UpSampling2D(size=(2, 2))(e3)

        e3 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=3, strides=(1, 1), 
                    kernel_initializer="he_normal", padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=3, strides=(1, 1),
                    kernel_initializer="he_normal", padding='same')(e3)
        e3 = ELU()(e3)

        e3 = UpSampling2D(size=(2, 2))(e3)

        e3 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=3, strides=(1, 1),
                    kernel_initializer="he_normal", padding='same')(e3)
        e3 = ELU()(e3)
        e3 = Conv2D(filters=int(self.conf.coarse_dim/2), kernel_size=3, strides=(1, 1),
                    kernel_initializer="he_normal", padding='same')(e3)
        e3out = ELU()(e3)
        

        # Decoder
        d = Concatenate(axis=3, name='input_decoder')([e1out, e2out, e3out])
        
        d = Conv2D(filters=int(self.conf.coarse_dim/8), kernel_size=3, strides=(1, 1),
                   kernel_initializer="he_normal", padding='same')(d)
        d = ELU()(d)
        reconstr_img = Conv2D(filters=int(self.conf.img_shape[2]), kernel_size=3, strides=(1, 1),
                         kernel_initializer="he_normal", padding='same', name='reconstructed_image')(d)
        

        return models.Model(inputs=[inputs_img, input_mask], outputs=[reconstr_img], name='Generator_A3E1D')


    def Unet(self):
        print('Create U-Net-2 Generator network...')

        def Conv2D_SubLayers(prev_layer, nr_filts, layer_name):
            # first layer
            a = Conv2D(filters=nr_filts, kernel_size=self.conf.filters, padding='same',
                       kernel_initializer="he_normal", name='%s_C1' %layer_name)(prev_layer)
            a = BatchNormalization(name='%s_BN1' %layer_name)(a)
            a = Activation("relu", name='%s_A1' %layer_name)(a)
            # second layer
            a = Conv2D(filters=nr_filts, kernel_size=self.conf.filters, padding='same',
                       kernel_initializer="he_normal", name='%s_C2' %layer_name)(a)
            a = BatchNormalization(name='%s_BN2' %layer_name)(a)
            a = Activation("relu", name='%s_A2' %layer_name)(a)
            return a

        img_input = Input(shape=self.conf.img_shape, name='Masked_Image')

        # U-Net Encoder - upper level
        e1c = Conv2D_SubLayers(prev_layer=img_input, nr_filts=int(self.conf.coarse_dim/8), layer_name='E1')
        e1 = MaxPooling2D(pool_size=(2, 2), name='E1_P')(e1c)
        e1 = Dropout(self.conf.dropout*0.5, name='E1_D2')(e1)

        # U-Net Encoder - second level
        e2c = Conv2D_SubLayers(prev_layer=e1, nr_filts=int(self.conf.coarse_dim/4), layer_name='E2')
        e2 = MaxPooling2D(pool_size=(2, 2), name='E2_P')(e2c)
        e2 = Dropout(self.conf.dropout, name='E2_D2')(e2)

        # U-Net Encoder - third level
        e3c = Conv2D_SubLayers(prev_layer=e2, nr_filts=int(self.conf.coarse_dim/2), layer_name='E3')
        e3 = MaxPooling2D(pool_size=(2, 2), name='E3_P')(e3c)
        e3 = Dropout(self.conf.dropout, name='E3_D2')(e3)  
                
        # U-Net Encoder - bottom level
        b = Conv2D_SubLayers(prev_layer=e3, nr_filts=int(self.conf.coarse_dim), layer_name='B')

        # U-Net Decoder - third level
        d3 = Conv2DTranspose(filters=int(self.conf.coarse_dim/2), kernel_size=self.conf.filters, 
                             strides=(2, 2), padding='same', name='D3_DC')(b)
        d3 = concatenate([d3, e3c], name='merge_layer_E3_A2')
        d3 = Dropout(self.conf.dropout, name='D3_D1')(d3)
        d3 = Conv2D_SubLayers(prev_layer=d3, nr_filts=int(self.conf.coarse_dim/2), layer_name='D3')

        # U-Net Decoder - second level
        d2 = Conv2DTranspose(filters=int(self.conf.coarse_dim/4), kernel_size=self.conf.filters, 
                             strides=(2, 2), padding='same', name='D2_DC')(d3)
        d2 = concatenate([d2, e2c], name='merge_layer_E2_A2')
        d2 = Dropout(self.conf.dropout, name='D2_D1')(d2)
        d2 = Conv2D_SubLayers(prev_layer=d2, nr_filts=int(self.conf.coarse_dim/4), layer_name='D2')

        # U-Net Decoder - upper level
        d1 = Conv2DTranspose(filters=int(self.conf.coarse_dim/8), kernel_size=self.conf.filters, 
                             strides=(2, 2), padding='same', name='D1_DC')(d2)
        d1 = concatenate([d1, e1c], name='merge_layer_E1_A2')
        d1 = Dropout(self.conf.dropout, name='D1_D1')(d1)
        d1 = Conv2D_SubLayers(prev_layer=d1, nr_filts=int(self.conf.coarse_dim/4), layer_name='D1')

        # Outro Layer
        output_image = Conv2D(filters=int(self.conf.img_shape[2]), kernel_size=self.conf.filters, 
                              strides=(1, 1), padding='same', activation='tanh', name='out_C')(d1)

        return models.Model(inputs=img_input, outputs=output_image, name='Generator_Unet')