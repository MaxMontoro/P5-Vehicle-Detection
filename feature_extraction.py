
from hog import get_hog_features, orient, pix_per_cell, cell_per_block, cspace, CONFIG
from utils import *

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='YUV', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    ''' Exctract features from a list of images '''
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for path in imgs:
        # Read in each one by one
        image = cv2.imread(path)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)

        # hog_features = get_hog_features(grayscale_img(image))

        # Append the new feature vector to the features list
        # features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        features.append(hog_features)

        # Return list of feature vectors
    return features


def extract_and_pickle_features():
    car_features = extract_features(vehicle_images, cspace=cspace, spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256))
    notcar_features = extract_features(non_vehicle_images, cspace=cspace, spatial_size=(32, 32),
                                       hist_bins=32, hist_range=(0, 256))

    with open(f'features/car_features{str(CONFIG)}.p', 'wb') as car_features_file:
        pickle.dump(car_features, car_features_file)

    with open(f'features/notcar_features{str(CONFIG)}.p', 'wb') as notcar_features_file:
        pickle.dump(notcar_features, notcar_features_file)
    return car_features, notcar_features


def load_features():
    with open(f'features/notcar_features{str(CONFIG)}.p', 'rb') as notcar_features_file:
        notcar_features = pickle.load(notcar_features_file)

    with open(f'features/car_features{str(CONFIG)}.p', 'rb') as car_features_file:
        car_features = pickle.load(car_features_file)
    return car_features, notcar_features


if __name__ == "__main__":
    car_features, notcar_features = extract_and_pickle_features()
    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(vehicle_images))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(vehicle_images[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.show()
    else:
        print('Your function only returns empty feature vectors...')
else:
    car_features, notcar_features = load_features()
