import cv2
import glob
import random
import pickle
import numpy as np

from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip


def read_image(fname):
    ''' Reads image from a path. The returned image is in RGB colorspace. '''
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def grayscale_img(img):
    ''' Return a grayscaled version of the image '''
    new_img = img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

vehicle_images = glob.glob('vehicles/**/*.png')
non_vehicle_images = glob.glob('non-vehicles/**/*.png')


if __name__ == '__main__':
    print(f'Vehicle images {len(vehicle_images)}')
    print(f'Non-vehicle images {len(non_vehicle_images)}')
    for i in range(15):
        img = cv2.imread(vehicle_images[random.randint(0, len(vehicle_images))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imsave(f'vehicle_examples/{i}.png', img)

        img = cv2.imread(non_vehicle_images[random.randint(0, len(non_vehicle_images))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # plt.imsave(f'non_vehicle_examples/{i}.png', img)
        plt.imshow(img)
        plt.show()
