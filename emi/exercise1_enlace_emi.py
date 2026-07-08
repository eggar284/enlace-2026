#import all libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, center_of_mass
#define our lenghts
width = 512
height = 140
#define variables for the functions in order to locate the files
folder = "rmot_absorption_images"
base_name = "2022_06_27_512x140_redMOT_shelving_(p=tof,exposure_time_side,number_of_images_side,exposure_time_vertical,number_of_images_vertical)#0.0_0.005_4_0.03_4_140957"
# function to extract the 3 files
def load_bin(suffix):
    path = os.path.join(folder, f"{base_name}_{suffix}.bin")
    data = np.fromfile(path, dtype=np.float64) #we transform our images into arrays in order to amnipulate them as numbers
    return data[:height*width].reshape(height, width) #takes only the first 71,680 numbers from the array and reshape it to have the image

#define our files storing them on a variable using the created function
I_atoms = load_bin(1)  #this image is in charge of showing the shadow caused thanks to the laser that it canot pass through the atoms
I_light = load_bin(2)  #the pure light withouth any atoms
I_background = load_bin(3) #this is an image that has no light or atoms so it tells us the error on the images which can be deleted by extracting this variable form the others

I_atoms_corrected = I_atoms - I_background # give the correction 
I_light_corrected = I_light - I_background # give the correction

#we use the gaussian filter to make an average of the neigbor pixels to see which ones weight mor or less this reduces the error on images
sigma = 2 #how far does the gaussian filter will look in his neigbor to determine the average
I_atoms_smooth = gaussian_filter(I_atoms_corrected, sigma=sigma) # we use a smoth gaussian crrection in order to have a best correction and reduce the error on  the images
I_light_smooth = gaussian_filter(I_light_corrected, sigma=sigma) # we use this on both variables in order to reduce the error

light_umbral = 0.15 * I_light_smooth.max() #get the 15% of the maximun value
valid_mask = I_light_smooth > light_umbral  #we use an operator to compare both values and see if it is a real pixel or one that is just making noisy our image

I_light_safe = np.where(valid_mask, I_light_smooth, 1e-10) #like an if conditional (for matrix) with the next structure : np.where(condition, value__fot, new_value) if value__fot is true then the variable specified on the first paremter adquires the new value we use that in order to define the values of our compartors as non valid or minimum
t = I_atoms_smooth / I_light_safe #our division for the formula
t[t <= 0] = 1e-10 #if t equals 0 or is less than 0 we substitute taht value by a super samll one

od = -np.log(t) #We calcultae the optical density with the negative value because of the formula
od = np.clip(od, 0, 5) # the clip has : np.clip(matrix, min, max) so we configure that max is always 5 and minimum is always 0
od[~valid_mask] = 0 #where light is not reliable we subsititute it by a 0

umbral_od = 0.5 * od.max() # we take max value of optical density and get the half of it
od_thresh = np.where(od > umbral_od, od, 0) # we get the condiftion that if (for matrix) optical density is bigger that the ubral substitute the variable od by 0, again this is to clear our matrix 
y_center, x_center = center_of_mass(od_thresh) #we calculate where the mass is distributed on a better way and get teh pixels wehere our atom is located
y_center, x_center = int(round(y_center)), int(round(x_center)) #the function of center of mass gives you float number so we need to store them as int values because we canot work with pixel matrixes using float values


half_y = 30 #we define our both variables of how much will we extend the space of our atom image 
half_x = 50 # we can also say that they are the cut ratios

y_min = np.clip(y_center - half_y, 0, height) #np.clip(value, 0, limit) we take a value and set min and max in array form
y_max = np.clip(y_center + half_y, 0, height)
x_min = np.clip(x_center - half_x, 0, width)
x_max = np.clip(x_center + half_x, 0, width)
#now we have formed a little squared with 4 varibles 

od_roi = od[y_min:y_max, x_min:x_max] #we slice the array (2d) in order to define our limits to focus only on the cloud of atoms

plt.figure(figsize=(8, 5)) # create our window
plt.imshow(od_roi, cmap='inferno', vmin=0, vmax=2, origin='lower') #deine what array will we graph, the color which is inferno, the vmin and vmax define the range of teh colors that we will use, the parameter origin deifnes how we will put our y axes or in other words defines where to place the (0,0)
plt.colorbar(label='Optical Density (OD)')
plt.title('ROI - Atomic Cloud')
plt.xlabel('X (px)')
plt.ylabel('Y (px)')
plt.tight_layout() #we adjust where elements are placed so it looks good
plt.show() #we show the graph