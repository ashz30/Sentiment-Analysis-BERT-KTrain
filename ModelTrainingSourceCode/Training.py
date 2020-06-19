
# copy pasted from my jupyter notebook, please solve any syntax errors as i havent run it post pasting

# install ktrain
#!pip3 install ktrain

#imports
import ktrain
from ktrain import text

print(ktrain.__version__)

# download IMDb movie review dataset
# for training any model, you need to have your own dataset, the more records the better
import tensorflow as tf
dataset = tf.keras.utils.get_file(
    fname="aclImdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True,
)


# set path to dataset
import os.path
#dataset = '/root/.keras/datasets/aclImdb'
IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')
print(IMDB_DATADIR)


trn, val, preproc = text.texts_from_folder(IMDB_DATADIR,
                                          maxlen=500,
                                          preprocess_mode='bert',
                                          train_test_names=['train',
                                                            'test'],
                                          classes=['pos', 'neg'])


model = text.text_classifier('bert', trn, preproc=preproc)
learner = ktrain.get_learner(model,train_data=trn, val_data=val, batch_size=6)

# Training - 1 epoch.
# to change training - increase number of epochs from 1 to higher and change learning rate
#Bert recommendations for fine-tuning: (not in the scope for the guide)
# Batch size: 16, 32
# Learning rate (Adam): 5e-5, 3e-5, 2e-5
# Number of epochs: 2, 3, 4

learner.fit_onecycle(2e-5, 1)

predictor = ktrain.get_predictor(learner.model, preproc)

#Predictor is used for predicting, save the model
predictor.save('<FolderLocation to save>')