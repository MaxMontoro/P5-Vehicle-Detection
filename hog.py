from utils import *

CONFIG = {'pix_per_cell': 16,
          'cell_per_block': 2,
          'orient': 11,
          'cspace': 'YUV'}

pix_per_cell = CONFIG.get('pix_per_cell')
cell_per_block = CONFIG.get('cell_per_block')
orient = CONFIG.get('orient')
cspace = CONFIG.get('cspace')

# Define a function to return HOG features and visualization
def get_hog_features(img, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
    return features


if __name__ == '__main__':
    car_images = glob.glob('vehicle_examples/*.png')
    for index, image_path in enumerate(car_images):
        img = grayscale_img(read_image(image_path))
        features, hog_image = get_hog_features(img, vis=True)
        plt.imsave(f'output_images/hog_{index}.png', hog_image)

    non_car_images = glob.glob('non_vehicle_examples/*.png')
    for index, image_path in enumerate(non_car_images):
        img = grayscale_img(read_image(image_path))
        features, hog_image = get_hog_features(img, vis=True)
        plt.imsave(f'output_images/non_vehicle_hog_{index}.png', hog_image)
