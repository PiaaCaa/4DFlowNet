import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#from matplotlib.animation import FuncAnimation
#from IPython import display
import matplotlib.pyplot as plt


#TODO check if correctly used
#TODO belongs into a different category
def check_and_normalize(img):
        if img.dtype == np.uint8:
                return np.asarray(img, dtype=float)/255

        return (img - np.min(img))/(np.max(img) - np.min(img))


#TODO extend to time 
def generate_gif_volume(img3D, axis = 0, save_as = "animation"):
    # check that input is 3 dimensional suc that normalization is correct
    img3D = img3D.squeeze()
    assert len(img3D.shape) == 3


    img3D = check_and_normalize(img3D)

    if axis == 0:
            frames = [Image.fromarray(img3D[i, :, :]*255) for i in range(img3D.shape[0])]
    elif axis ==1:
            frames = [Image.fromarray(img3D[:, i, :]*255) for i in range(img3D.shape[1])]
    elif axis == 2:
            frames = [Image.fromarray(img3D[:, :, i]*255) for i in range(img3D.shape[2])]
    else: 
        print("Invalid axis input.")
    
    frame_one = frames[0]
    frame_one.save("/home/pcallmer/Temporal4DFlowNet/results/plots" +save_as+".gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0) #/home/pcallmer/Temporal4DFlowNet/results/plots





