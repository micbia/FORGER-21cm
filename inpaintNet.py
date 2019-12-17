import numpy as np, os, sys, configparser, matplotlib.pylab as plt

from time import time, strftime, gmtime
from datetime import datetime

from keras import backend as K
from keras import models, optimizers
from keras.utils import multi_gpu_model, plot_model
from keras.backend import set_value

from config.net_config import NetworkConfig
from model.networks import NetworkComponents

from utils.other_utils import GenerateNoise, GenerateLabels, RescaleData
from utils.load import LoadData
from utils.plotting import Plots
from utils.metrics import wasserstein_loss, perceptual_loss, UnFreezeNetwork


class InpaintNetwork:
    def __init__(self, CONFIG_FILE='./config/example.ini', PATH_OUTPUT=None):
        print('GAN network')

        # Configure networks
        self.config_file = CONFIG_FILE
        self.conf = NetworkConfig(self.config_file)
        self.optimizer = optimizers.Adam(lr=2e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)   # see Radford et al. 2015

        self.nr_discr = self.conf.Dsize

        if('wasserstein' in self.conf.lossD):
            self.lossD = wasserstein_loss
        else:
            self.lossD = 'binary_crossentropy'

        self.lossGAN = self.conf.lossGAN
        self.wlossGAN = self.conf.wlossGAN

        if('wasserstein' in self.lossGAN):
            print('Wasserstein loss for discriminator in GAN network')
            self.lossGAN = [self.lossGAN[0], wasserstein_loss]

        # Create output directory and sub-directories
        if PATH_OUTPUT != None:
            self.path_output = PATH_OUTPUT + datetime.now().strftime('%d-%mT%H-%M-%S')
            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                os.makedirs(self.path_output+'/model')
                os.makedirs(self.path_output+'/images')
                os.makedirs(self.path_output+'/images/generated_test')
                os.makedirs(self.path_output+'/images/pdf_gan')
                os.makedirs(self.path_output+'/checkpoints')
                os.makedirs(self.path_output+'/checkpoints/weights')
        else:
            self.path_output = datetime.now().strftime('%d-%mT%H-%M-%S')
            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                os.makedirs(self.path_output+'/model')
                os.makedirs(self.path_output+'/images')
                os.makedirs(self.path_output+'/images/generated_test')
                os.makedirs(self.path_output+'/images/pdf_gan')
                os.makedirs(self.path_output+'/checkpoints')
                os.makedirs(self.path_output+'/checkpoints/weights')
        self.path_output += '/'

        # copy ini file into output directory
        os.system('cp %s %s' %(self.config_file, self.path_output+'/model'))


    def GAN(self):
        net = NetworkComponents(CONFIG_FILE=self.config_file, PATH_OUTPUT=self.path_output)
        
        # built generator networks
        if(self.conf.type_of_gen == 'auto'):
            self.generator = net.Autoencoder()
        elif(self.conf.type_of_gen == 'auto31'):
            self.generator = net.Auto3Encoder1Decoder()
        elif(self.conf.type_of_gen == 'unet'):
            self.generator = net.Unet()
        
        # built adversary network
        if(self.nr_discr == 1):
            self.discriminator = net.Discriminator()
        else:
            self.globaldiscriminator = net.GlobalDiscriminator()
            self.localdiscriminator = net.LocalDiscriminator()
        
            
        # resume models weights from last checkpoint 
        if(self.conf.resume_path != None and self.conf.resume_epoch != 0):
            self.generator.load_weights('%sweights/model_weights-ep%d.h5' %(self.conf.resume_path, self.conf.resume_epoch))
            self.discriminator.load_weights('%sweights/model_weights-ep%d.h5' %(self.conf.resume_path, self.conf.resume_epoch))
            print('Adversary and Generator Model weights resumed from:\tmodel_weights-ep%d.h5' %(self.conf.resume_epoch))
            
            # copy logs checkpoints
            os.system('cp %s*ep-%d.txt %scheckpoints/' %(self.conf.resume_path, self.conf.resume_epoch, self.path_output))
        else:
            print('GAN Model Created')

        # freeze discriminator and its layers, un-freeze generator and its layers
        UnFreezeNetwork(net=self.generator, state=True)
        if(self.nr_discr == 1):
            UnFreezeNetwork(net=self.discriminator, state=False)
        else:
            UnFreezeNetwork(net=self.globaldiscriminator, state=False)
            UnFreezeNetwork(net=self.localdiscriminator, state=False)
        
        # Create GAN network by stucking the generator and adversary network
        input_img = models.Input(shape=self.conf.img_shape, name='input_img')
        input_mask = models.Input(shape=self.conf.img_shape, name='input_mask')
        reconstr_img = self.generator([input_img, input_mask])           # generator layers
        
        if(self.nr_discr == 1):
            label_output = self.discriminator(reconstr_img)     # discriminator layers

            self.gan = models.Model([input_img, input_mask], label_output, name='GAN')
            self.gan.compile(loss=self.lossGAN, optimizer=self.optimizer)
        else:
            global_label_output = self.globaldiscriminator(reconstr_img)
            local_label_output = self.localdiscriminator([reconstr_img, input_mask])

            self.gan = models.Model([input_img, input_mask], [global_label_output, local_label_output], name='GAN')
            self.gan.compile(loss=self.lossGAN, loss_weights=self.wlossGAN, optimizer=self.optimizer)

        """        
        if(len(self.lossGAN) == 1 and len(self.wlossGAN) == 1):
            self.gan = models.Model([input_img, input_mask], label_output)
            self.gan.compile(loss=self.lossGAN, optimizer=self.optimizer)
        else:
            self.gan = models.Model([input_img, input_mask], [reconstr_img, label_output])
            self.gan.compile(loss=self.lossGAN, loss_weights=self.wlossGAN, optimizer=self.optimizer)
        """

        # un-freeze discriminator and its layers, freeze generator and its layers
        UnFreezeNetwork(net=self.generator, state=False)
        if(self.nr_discr == 1):
            UnFreezeNetwork(net=self.discriminator, state=True)
            # compile discriminator
            self.discriminator.compile(loss=self.lossD, optimizer=self.optimizer)
        else:
            UnFreezeNetwork(net=self.globaldiscriminator, state=True)
            UnFreezeNetwork(net=self.localdiscriminator, state=True)
            # compile discriminators
            self.globaldiscriminator.compile(loss=self.lossD, optimizer=self.optimizer)
            self.localdiscriminator.compile(loss=self.lossD, optimizer=self.optimizer)


        # Save model visualization
        plot_model(self.generator, to_file=self.path_output+'model/generator_visualization.png', show_shapes=True, show_layer_names=True)
        if(self.nr_discr == 1):
            plot_model(self.discriminator, to_file=self.path_output+'model/adversary_visualization.png', show_shapes=True, show_layer_names=True)
        else:
            plot_model(self.globaldiscriminator, to_file=self.path_output+'model/global_adversary_visualization.png', show_shapes=True, show_layer_names=True)
            plot_model(self.localdiscriminator, to_file=self.path_output+'model/local_adversary_visualization.png', show_shapes=True, show_layer_names=True)
        plot_model(self.gan, to_file=self.path_output+'model/gan_visualization.png', show_shapes=True, show_layer_names=True)
        return self.gan
        

    def CreateCheckpoint(self, epch, prev_epch):
        self.generator.save_weights('%scheckpoints/weights/generator_weights_ep-%d.h5' %(self.path_output, epch))
        self.discriminator.save_weights('%scheckpoints/weights/adversary_weights_ep-%d.h5' %(self.path_output, epch))

        # delete previous losses checkpoint
        if(prev_epch != 0):
            os.remove('%scheckpoints/lossG_ep-%d.txt' %(self.path_output, prev_epch))
            os.remove('%scheckpoints/lossA_ep-%d.txt' %(self.path_output, prev_epch))
            os.remove('%scheckpoints/lossAr_ep-%d.txt' %(self.path_output, prev_epch))
            os.remove('%scheckpoints/lossAf_ep-%d.txt' %(self.path_output, prev_epch))

        # save new losses checkpoint
        np.savetxt('%scheckpoints/lossG_ep-%d.txt' %(self.path_output, epch), self.loss_G)
        np.savetxt('%scheckpoints/lossAr_ep-%d.txt' %(self.path_output, epch), self.loss_D_real)
        np.savetxt('%scheckpoints/lossAf_ep-%d.txt' %(self.path_output, epch), self.loss_D_fake)


    def TrainGAN(self):
        # variable dataset will be the directory containing the trianing data
        ld = LoadData(IMG_SHAPE=self.conf.img_shape, PATH_DATA=self.conf.dataset, PATH_MASK='inputs/mask/testing_mask_dataset/')
        plot = Plots(IMG_SHAPE=self.conf.img_shape, PATH_DATA=self.conf.dataset, PATH_MASK='inputs/mask/testing_mask_dataset/', PATH_OUTPUT=self.path_output)

        if(len(self.lossGAN) == 1 and len(self.wlossGAN) == 1):
            self.loss_G = []
        else:
            self.loss_G = []
            self.loss_G1 = []
            self.loss_G2 = []

        self.loss_D_real = []
        self.loss_D_fake = []

        self.lr = []
        prev_epoch = 0

        for ep in range(self.conf.epochs):
            t1 = time()

            if((ep+1)%self.conf.lr_decay == 0 and (ep+1) < 1000):
                # reduce leaning rate
                lr = K.get_value(self.gan.optimizer.lr)
                lr *= 0.5
                set_value(self.gan.optimizer.lr, lr)
                
                if(self.nr_discr == 1):
                    set_value(self.discriminator.optimizer.lr, lr)
                else:
                    set_value(self.globaldiscriminator.optimizer.lr, lr)
                    set_value(self.localdiscriminator.optimizer.lr, lr)

            for bt in range(self.conf.batch_size):
                # un-freeze discriminator, freze generator
                UnFreezeNetwork(net=self.generator, state=False)
                if(self.nr_discr == 1):
                    UnFreezeNetwork(net=self.discriminator, state=True)
                else:
                    UnFreezeNetwork(net=self.globaldiscriminator, state=True)
                    UnFreezeNetwork(net=self.localdiscriminator, state=True)
            
                # train discriminator more then generator
                for k in range(1):
                    # create batch of real and masked images
                    real_images, masksed_images, masks = ld.LoadMaskedData(batch=self.conf.batch_size)

                    # generator reconstructs the missing part in masked images
                    fake_images = self.generator.predict([real_images, masks])
                
                    # generate smooth label, with 5% of indexes flipped
                    real_label, fake_label = GenerateLabels(self.conf.batch_size)

                    # train adversary network, for separated mini-batchs, see Ioffe et al. 2015
                    if(self.nr_discr == 1):
                        loss_real = self.discriminator.train_on_batch(real_images, real_label)
                        loss_fake = self.discriminator.train_on_batch(fake_images, fake_label)
                    else:
                        glob_loss_real = self.globaldiscriminator.train_on_batch(real_images, real_label)
                        glob_loss_fake = self.globaldiscriminator.train_on_batch(fake_images, fake_label)
                        loc_loss_real = self.localdiscriminator.train_on_batch([real_images, masks], real_label)
                        loc_loss_fake = self.localdiscriminator.train_on_batch([fake_images, masks], fake_label)

                        loss_real = self.wlossGAN[0]*glob_loss_real + self.wlossGAN[1]*loc_loss_real
                        loss_fake = self.wlossGAN[0]*glob_loss_fake + self.wlossGAN[1]*loc_loss_fake

                # freeze discriminator, un-freeze generator
                UnFreezeNetwork(net=self.generator, state=True)
                if(self.nr_discr == 1):
                    UnFreezeNetwork(net=self.discriminator, state=False)
                else:
                    UnFreezeNetwork(net=self.globaldiscriminator, state=False)
                    UnFreezeNetwork(net=self.localdiscriminator, state=False)

                if(len(self.lossGAN) == 1 and len(self.wlossGAN) == 1 or self.nr_discr == 1):
                    real_label2 = GenerateLabels(self.conf.batch_size, return_label='real')
                    loss_gan = self.gan.train_on_batch(masksed_images, real_label2)            
                else:
                    global_real_label = GenerateLabels(self.conf.batch_size, return_label='real')
                    local_real_label = GenerateLabels(self.conf.batch_size, return_label='real')
                    loss_gan = self.gan.train_on_batch([real_images, masks], [global_real_label, local_real_label])

            # store losses at the end of every batch cycle
            self.loss_D_real.append(loss_real)
            self.loss_D_fake.append(loss_fake)
            if(len(self.lossGAN) == 1 and len(self.wlossGAN) == 1 or self.nr_discr == 1):
                self.loss_G.append(loss_gan)
            else:
                self.loss_G.append(loss_gan[0])
                self.loss_G1.append(loss_gan[1])
                self.loss_G2.append(loss_gan[2])
            
            # networks prediction for PDF
            if(self.nr_discr == 1):
                distr_real = self.discriminator.predict(real_images).T[0]
                distr_fake = self.discriminator.predict(fake_images).T[0]
            else:
                global_distr_real = self.globaldiscriminator.predict(real_images).T[0]
                global_distr_fake = self.globaldiscriminator.predict(fake_images).T[0]
                local_distr_real = self.localdiscriminator.predict([real_images, masks]).T[0]
                local_distr_fake = self.localdiscriminator.predict([fake_images, masks]).T[0]
            
            distr_reconst = self.gan.predict([real_images, masks])[1].T[0]


            # Create weight checkpoint and intermediate plots
            if(ep == 0 or (ep+1) % 1 == 0 or (ep+1) == self.conf.epochs):
                #self.CreateCheckpoint(epch=ep, prev_epch=prev_epoch)
                plot.PlotLosses(epch=ep, prev_epch=prev_epoch, loss_D_real=self.loss_D_real, loss_D_fake=self.loss_D_fake, loss_G=self.loss_G, lr=self.lr)
                
                if('mnist' in self.conf.dataset):
                    plot.PlotSpecificMNISTImages(epch=ep, gmodel=self.generator)
                else:
                    plot.PlotSpecificImages(epch=ep, gmodel=self.generator)
                    if(self.nr_discr == 1):
                        plot.PlotDistribution(ep ,fake_label, distr_fake, real_label, distr_real, real_label2, distr_reconst)
                    else:
                        plot.PlotDistribution(ep ,fake_label, global_distr_fake, real_label, global_distr_real, global_real_label, distr_reconst)
                        plot.PlotDistribution(ep ,fake_label, local_distr_fake, real_label, local_distr_real, local_real_label, distr_reconst)
                        
                prev_epoch = ep

            t2 = time()
            if(len(self.lossGAN) == 1 and len(self.wlossGAN) == 1):
                print(' Epoch %d : t=%3ds  ---  [ D: L_real=%.2f, L_fake=%.2f ]  [ G: L_gan=%.2f ]' %(ep+1, t2-t1, self.loss_D_real[ep], self.loss_D_fake[ep], self.loss_G[ep]))
            else:
                print(' Epoch %d : t=%3ds  ---  [ D: L_real=%.2f, L_fake=%.2f ]  [ G: L_gan=%.2f, L1=%.2f, L2=%.2f ]' %(ep+1, t2-t1, self.loss_D_real[ep], self.loss_D_fake[ep], self.loss_G[ep], self.loss_G1[ep], self.loss_G2[ep]))
            
        # save final losses and weights
        if(len(self.lossGAN) == 1 and len(self.wlossGAN) == 1):
            np.savetxt('%slossG.txt' %self.path_output, self.loss_G)
        else:
            np.savetxt('%slossG.txt' %self.path_output, self.loss_G)
            np.savetxt('%slossG1.txt' %self.path_output, self.loss_G1)
            np.savetxt('%slossG2.txt' %self.path_output, self.loss_G2)
    
        np.savetxt('%slossAr.txt' %self.path_output, self.loss_D_real)
        np.savetxt('%slossAf.txt' %self.path_output, self.loss_D_fake)
        np.savetxt('%slr.txt' %self.path_output, self.lr)

        #self.generator.save_weights('%smodel/generator_weights.h5' %self.path_output)
        #self.discriminator.save_weights('%smodel/adversary_weights.h5' %self.path_output)

        return 1
