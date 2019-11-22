import numpy as np, random

def ReadTensor(filename, bits=32, order='C', dimensions=3):
    ''' Read a binary file with three inital integers (a cbin file).

    Parameters:
        * filename (string): the filename to read from
        * bits = 32 (integer): the number of bits in the file
        * order = 'C' (string): the ordering of the data. Can be 'C'
            for C style ordering, or 'F' for fortran style.
        * dimensions (int): the number of dimensions of the data (default:3)

    Returns:
        The data as a three dimensional numpy array.
    '''

    assert(bits == 32 or bits == 64)

    f = open(filename)

    temp_mesh = np.fromfile(f, count=dimensions, dtype='int32')

    datatype = np.float32 if bits == 32 else np.float64
    data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
    data = data.reshape(temp_mesh, order=order)
    return data


def GenerateNoise(batch_size, input_size):
    return np.random.normal(loc=0., scale=1., size=[batch_size, input_size])


def SwapElements(x, y, pflip): 
    ''' x and y must be 1D-array of the same size ''' 
    nr_flip = int(pflip*x.shape[0]) 
    idx = random.choices(range(x.size), k=nr_flip) 

    dummy = x[idx] 
    for i in range(len(idx)): 
        x[i] = y[i] 
        y[i] = dummy[i]  
    return x, y


def GenerateLabels(batch_size, return_label='both'):
    ''' Generate smooth label, considered as additional noise.
        it has been proven that train better network (Salimans et al. 2016).
        Also randomly flip 5% of the labels seams to work. '''

    real_label, fake_label = SwapElements(x=np.random.uniform(0.7, 1.2, size=batch_size), 
                                          y=np.random.uniform(0., 0.3, size=batch_size), 
                                          pflip=0.05)

    if(return_label == 'fake'):
        return fake_label
    elif(return_label == 'real'):
        return real_label
    else:
        return real_label, fake_label


def RescaleData(arr, a=-1, b=1):
    scaled_arr = (arr.astype(np.float32) - np.min(arr))/(np.max(arr) - np.min(arr)) * (b-a) + a
    return scaled_arr