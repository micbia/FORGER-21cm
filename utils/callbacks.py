import numpy as np
from keras import callbacks

class HistoryCheckpoint(callbacks.Callback):
    def __init__(self, filepath='./', verbose=0, save_freq=1, in_epoch=0):
        self.verbose = verbose
        self.filepath = filepath
        self.save_freq = save_freq
        self.stor_arr = []
        self.prev_epoch = 0
        self.in_epoch = in_epoch

    def on_train_begin(self, logs=None):
        if(self.in_epoch != 0):
            print('Resuming from Epoch %d...' %self.in_epoch)
            self.prev_epoch = self.in_epoch

    def on_epoch_end(self, epoch, logs=None):
        if(epoch == self.in_epoch): self.stor_arr =  [[] for i in range(len(logs))]     # initializate array
        
        fname = self.filepath+'%s_ep-%d.txt'

        if(epoch % self.save_freq == 0 and epoch != self.in_epoch): 
            for i, val in enumerate(logs):
                self.stor_arr[i] = np.append(self.stor_arr[i], logs[val])
                if(os.path.isfile(fname %(val, self.prev_epoch))):
                    chekp_arr = np.loadtxt(fname %(val, self.prev_epoch)) # load previous save
                    chekp_arr = np.append(chekp_arr, self.stor_arr[i])      # update 
                    np.savetxt(fname %(val, epoch), chekp_arr)            # save
                    os.remove(fname %(val, self.prev_epoch))              # delete old save
                else:
                    np.savetxt(fname %(val, epoch), self.stor_arr[i])
            
            self.prev_epoch = epoch
            self.stor_arr = [[] for i in range(len(logs))]          # empty storing array

            if(self.verbose): print('Updated Logs checkpoints for epoch %d.' %epoch)
        else:
            for j, val in enumerate(logs):
                self.stor_arr[j] = np.append(self.stor_arr[j], logs[val])
