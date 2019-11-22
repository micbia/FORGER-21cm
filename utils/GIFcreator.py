import numpy as np, os
from glob import glob
from moviepy.editor import ImageSequenceClip



def CreateGif(filename, array, fps=5, scale=1., fmt='gif'):
    ''' Create and save a gif or video from array of images.
        Parameters:
            * filename (string): name of the saved video
            * array (list or string): array of images name already in order, if string it supposed to be the first part of the images name (before iteration integer)
            * fps = 5 (integer): frame per seconds (limit human eye ~ 15)
            * scale = 1. (float): ratio factor to scale image hight and width
            * fmt (string): file extention of the gif/video (e.g: 'gif', 'mp4' or 'avi')
        Return:
            * moviepy clip object
    '''
    if(isinstance(array, str)):
        array = np.array(sorted(glob(array+'*.png'), key=os.path.getmtime))
    else:
        pass
    filename += '.'+fmt
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    if(fmt == 'gif'):
        clip.write_gif(filename, fps=fps)
    elif(fmt == 'mp4'):
        clip.write_videofile(filename, fps=fps, codec='mpeg4')
    elif(fmt == 'avi'):
        clip.write_videofile(filename, fps=fps, codec='png')
    else:
        print('Error! Wrong File extension.')
        sys.exit()
    command = os.popen('du -sh %s' % filename)
    print(command.read())
    return clip
