import sys
from time import time, strftime, gmtime
from inpaintNet import InpaintNetwork

script, init_file = sys.argv

t_start = time()

net = InpaintNetwork(CONFIG_FILE=init_file)
net.GAN()           # create network and compile
net.TrainGAN()      # train network

t_end = time()
t_elapsed = strftime("%Hh %Mm %Ss", gmtime(t_end - t_start))
print('\nElapsed time to run script %s is:\t%s' %(script, t_elapsed))