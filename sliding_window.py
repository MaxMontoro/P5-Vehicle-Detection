from utils import *
import time

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list, weight=1):
    # Iterate through list of bboxes
    for index, box in enumerate(bbox_list):
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += weight
    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold=3):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

from classifier import svc, X_scaler
from hog import get_hog_features, orient, pix_per_cell, cell_per_block

img = mpimg.imread('test1.jpg')

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc,
              X_scaler=X_scaler,
              orient=orient,
              pix_per_cell=pix_per_cell,
              cell_per_block=cell_per_block,
              visualize_all_boxes=False):

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1, ch2, ch3 = ctrans_tosearch[:,:,0], ctrans_tosearch[:,:,1], ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1, hog2, hog3 = (get_hog_features(channel, orient, pix_per_cell,
                                         cell_per_block, feature_vec=False)
                        for channel in (ch1, ch2, ch3))

    rectangles = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos, xpos = yb*cells_per_step, xb*cells_per_step
            # Extract HOG for this patch
            hog_features = np.hstack(hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                                     for hog in (hog1, hog2, hog3))
            xleft, ytop = xpos*pix_per_cell, ypos*pix_per_cell

            test_features = X_scaler.transform(hog_features.reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1 or visualize_all_boxes:
                xbox_left, ytop_draw = np.int(xleft*scale), np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangle = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                rectangles.append(rectangle)
    return rectangles

scales =   (    1,          1.2,        1.5,       1.6,        2,         2.5,       3,          3.5)
y_values = ((400, 470), (410, 490), (400, 500), (430, 530),(440, 580),(410, 700),(430, 680), (420, 660))


def find_cars_on_image(img, scales=scales, y_values=y_values):
    box_list = []
    for scale, position in zip(scales, y_values):
        boxes = find_cars(img, position[0], position[1], scale, svc, visualize_all_boxes=False)
        box_list.extend(boxes)
    return box_list

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        box_ratio =  ( bbox[1][0]-bbox[0][0] ) / ( bbox[1][1] - bbox[0][1])
        if .3 < box_ratio <  3:  
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

class Cache:
    '''
    Helper class to keep track of the last few frames
    '''
    def __init__(self):
        self.boxes = []

    def add(self, boxes):
        self.boxes.append(boxes)
        if len(self.boxes) > 6:
            self.boxes = self.boxes[1:]

CACHE = Cache()

import random

def validate_boxes(boxes):
    return [box for box in boxes if box[0][0] not in range(520,750)]

def process_image(img):
    ''' Main image processing pipeline '''
    boxes = find_cars_on_image(img)
    boxes = validate_boxes(boxes)
    CACHE.add(boxes)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    for index, previous_boxes in enumerate(CACHE.boxes[::-1]):
        heat = add_heat(heat, previous_boxes)
    heat = apply_threshold(heat, 12)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    if random.randint(1,20) == 3:
        plt.imsave(f'output_images/{str(time.time())}.png', draw_img)
    return draw_img


if __name__ == "__main__":
    images = glob.glob('test_images/*')
    for path in images:
        image = read_image(path)
        boxes = find_cars_on_image(image)
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = add_heat(heat, boxes)
        heat = apply_threshold(heat, 2)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.imsave(f"output_images/label_{path.split('/')[1]}.png", labels[0], cmap='gray')
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.savefig(f"output_images/_{path.split('/')[1]}.png", )
        #plt.show()
