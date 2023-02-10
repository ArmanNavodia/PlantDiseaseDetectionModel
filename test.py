#%%
# Get test data
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import matplotlib.image as mpimg
import urllib
import time
import urllib.request
# from PIL import Image


async def download_image(image_url):
    f = open('./test/Test.jpg','wb')
    f.write(urllib.request.urlopen(image_url).read())
    time.sleep(2)
    f.close()
    
    

    

#%%
# load model
model = load_model('PlantModel.h5')
#%%
test_path = './test'

#%%
test_imgs = [os.path.join(test_path,img) for img in os.listdir(test_path)]
df_test = pd.DataFrame(test_imgs, columns=['Path'])

df_test.head()

#%%
#Create Test Dataset
IMG_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
OUTPUT_SHAPE = 38
NUM_EPOCHS = 10
#%%
def create_info_df(path):
    """
    input: `path` - folder path
    From folder path, create a Dataframe with columns: 
    Plant | Category | Path | Plant___Category | Disease
    return DataFrame
    """

    list_plants = []
    list_dir = os.listdir(path) # Get list direcotry
    # Go through each folder to create url and get required information
    # for plant in list_dir:
        # url = path +'/'+plant
        # for img in os.listdir(url):
        #     list_plants.append([*plant.split('___'), url+'/'+img, plant])
        # list_plants.append("")


    # Create DataFrame
    df = pd.DataFrame(list_plants, columns=['Plant', 'Category', 'Path','Plant___Category'])
    # Add `Disease` column - if folder name is not Healthy then plant is diseased
    df['Disease'] = df.Category.apply(lambda x: 0 if x=='healthy' else 1)

    return df
#%%
def decode_img(path, img_shape=IMG_SHAPE):
    """
    Read image from `path`, and convert the image to a 3D tensor
    return resized image.
    input: `path`: Path to an image
    return: resized tensor image
    """
    print('Image size: ({})'.format(img_shape))
    # Read the image file
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)/255
    # Resize image to our desired size
    img = tf.image.resize(img, img_shape)
    return img
#%%
def configure_for_performance(ds):
    #ds = ds.cache()
    ds = ds.batch(batch_size)
    #ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
#%%
# def get_pred_label(prediction_probabilities):
#     """
#     Turns an array of prediction probabilities into a label.
#     """
#     return unique_plant_cat[np.argmax(prediction_probabilities)]
#%%
def create_dataset(X, y=None, valid_data=False, test_data=False, img_shape=IMG_SHAPE):
    """
    Create Dataset from Images (X) and Labels (y)
    Shuffles the data if it's training data but doesn't shuffle if it validation data.
    Also accepts test data as input (no labels).
    Return Dataset 
    """
    print("Creating data set...")
    # If test data, there is no labels
    if test_data:       
        print("Creating test data batches...")
        dataset = tf.data.Dataset.from_tensor_slices((X))
        dataset = dataset.map(lambda x: decode_img(x, img_shape), num_parallel_calls=AUTOTUNE)
        dataset = configure_for_performance(dataset)
   
    print(dataset.element_spec)

    return dataset
#%%
# Create Models function utils #
################################

# Callbacks
# Early stopping Callbacks
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)

# Reduce Learning rate Callbacks
# lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
#                                                    patience=3,
#                                                    factor=0.2,
#                                                    verbose=2,
#                                                    mode='min')

#%%
def prediction():
    
    test_dataset = create_dataset(df_test['Path'], test_data=True, img_shape=IMG_SHAPE)
    # train_path = ''
    # train_info = create_info_df(train_path)
    # #Unique label list:
    # unique_plant_cat = np.unique(train_info['Plant___Category'].to_numpy())

    # Create a DF with Predictions
    # preds_df = pd.DataFrame(columns=["id"] + list(unique_plant_cat))
    # Append test image ID's to prediction DataFrame
    test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
    # preds_df["id"] = test_ids
    test_preds = model.predict(test_dataset)
    # Add the prediction probabilities to each plants category columns
    # preds_df[list(unique_plant_cat)] = test_preds
    # preds_df.head()
    # Show Test Images
    images_test = []
    for img_path in df_test['Path']:
        images_test.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,40))
    for i, image in enumerate(images_test):
        plt.subplot(11,3,i+1)
        plt.imshow(image)
        plt.title('Pred: {} - {:2.0f}%'.format('probability of disease:', np.max(test_preds[i])*100))
        plt.xlabel('image captured', fontsize=13, color='blue')
        plt.xticks([])
        plt.yticks([])
        print('probability of disease:', np.max(test_preds[i])*100)







