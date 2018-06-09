import cv2
import json
import numpy as np
import os
import pandas as pd
import random
import scipy.misc
import time
import utils

from IPython.display import display
from PIL import Image
from scipy import integrate
from scipy import misc
from scipy import stats
from skimage import img_as_float
from skimage import io
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# GLOBAL CONSTANTS
GRID_RANGE = [65, 20, 144, -50] #added this
DATA_FILE = 'leafsnap-dataset-images.csv'
NUM_CLASSES = 185
bad_lab_species = set(['Abies concolor', 'Abies nordmanniana', 'Picea pungens', 'Picea orientalis',
                       'Picea abies', 'Cedrus libani', 'Cedrus atlantica', 'Cedrus deodara',
                       'Juniperus virginiana', 'Tsuga canadensis', 'Larix decidua', 'Pseudolarix amabilis'])

columns = ['file_id', 'image_pat', 'segmented_path', 'species', 'source']
data = pd.read_csv(DATA_FILE, names=columns, header=1)

# Drop bad image samples
bad_indices = []
bad_lab_species_pattern = '|'.join(bad_lab_species)
for i in range(len(data)):
    if ((data.get_value(i, 'species') in bad_lab_species_pattern) and (data.get_value(i, 'source').lower() in 'lab')):
        bad_indices.append(i)
data.drop(data.index[bad_indices], inplace=True)

# Split train test
train_df = data.sample(frac=0.80, random_state=7)
test_df = data.drop(train_df.index)

images_train_original = train_df['image_pat'].tolist()
images_train_segmented = train_df['segmented_path'].tolist()
images_train = {'original': images_train_original, 'segmented': images_train_segmented}
species_train = train_df['species'].tolist()
species_classes_train = sorted(set(species_train))

images_test_original = test_df['image_pat'].tolist()
images_test_segmented = test_df['segmented_path'].tolist()
images_test = {'original': images_test_original, 'segmented': images_test_segmented}
species_test = test_df['species'].tolist()
species_classes_test = sorted(set(species_test))

print('\n[INFO]  Training Samples : {:5d}'.format(len(images_train['original'])))
print('\tTesting Samples  : {:5d}'.format(len(images_test['original'])))

print('[INFO] Processing Images')

def grid_loc(species, b_channel):
  #assumes dictionary is built and in scope, and make_grid has been called
  coords = dict[species]
  latMax = int(coords[0])
  latMin = int(coords[1])
  lonMin = int(coords[2])
  lonMax = int(coords[3])
  #get rand cell
  latR = random.randint(latMin, latMax)
  lonR = random.randint(lonMin, lonMax)
  #choose cell
  lat_step = (GRID_RANGE[0] - GRID_RANGE[1])/b_channel.shape[0] #assume square img
  lon_step = (GRID_RANGE[2] - GRID_RANGE[3])/b_channel.shape[0]
  c = latR/lat_step
  r = lonR/lon_step
  grid = np.zeros(b_channel.shape, dtype=b_channel.dtype)
  grid[r][c] = 1
  return grid
  
  
def save_images(images, species, directory='train', csv_name='temp.csv', augment=False, add_geolocate = True):
    cropped_images = []
    image_species = []
    image_paths = []
    count = 1
    write_dir = 'dataset/{}'.format(directory)
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    for index in range(len(images['original'])):
        image = utils.load_image_and_preprocess(
            images['original'][index], images['segmented'][index])
        if type(image) != type([]):
            image_dir = '{}/{}'.format(write_dir, species[index].lower().replace(' ', '_'))
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)

            file_name = '{}.jpg'.format(count)

            image_to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(image_dir, file_name), image_to_write)
            image_paths.append(os.path.join(image_dir, file_name))
            cropped_images.append(image)
            image_species.append(species[index])
            count += 1
            
            if add_geolocate: #need to remember later on in file that img new dim, and adjust layers 1
              b_channel, g_channel, r_channel = cv2.split(image)
              alpha_channel = grid_loc(species[index], b_channel) #b_channel just to ensure correct dim
              image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) #OVERWRITES original
              

            if augment:
                angle = 90
                while angle < 360:
                    rotated_image = utils.rotate(image, angle)

                    file_name = '{}.jpg'.format(count)
                    image_to_write = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)
                    #cv2.imwrite(os.path.join(image_dir, file_name), image_to_write)
                    result = Image.fromarray((image_to_write).astype(np.uint8))
                    result.save(os.path.join(image_dir, file_name))
                    image_paths.append(os.path.join(image_dir, file_name))
                    cropped_images.append(rotated_image)
                    image_species.append(species[index])

                    angle += 90
                    count += 1

        if index > 0 and index % 1000 == 0:
            print('[INFO] Processed {:5d} images'.format(index))

    print('[INFO] Final Number of {} Samples: {}'.format(directory, len(image_paths)))
    raw_data = {'image_paths': image_paths,
                'species': image_species}
    # df = pd.DataFrame(raw_data, columns = ['image_paths', 'species'])
    # df.to_csv(csv_name)

save_images(images_train, species_train, directory='train',
            csv_name='leafsnap-dataset-train-images.csv', augment=True)
save_images(images_test, species_test, directory='test',
            csv_name='leafsnap-dataset-test-images.csv', augment=False)

print('\n[DONE]')
