import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz" # grab data set from link if we've not already got it
dataset = tf.keras.utils.get_file("aclImdb_v1", url, # this ear marks the entire folder/file structure from above that we just got 
  untar=True, cache_dir='.',
  cache_subdir='')


dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train') # create a string variable that combines the directory and the sub folder 'train' 
shutil.rmtree(os.path.join(train_dir, 'unsup')) # remove the directory created by the inner function 


# batch_size = 32
# seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory( #use a text data set, data comes from the first param
  'aclImdb/train', 
  batch_size=32, 
  validation_split=0.2, 
  subset='training', 
  seed=42)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
  'aclImdb/train', 
  batch_size=32, 
  validation_split=0.2, 
  subset='validation', 
  seed=42)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
  'aclImdb/test', 
  batch_size=32)

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data) # convert all of the reviews to lowercase
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ') #strip all <br>'s from the string and replace with ' ' 
  return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation),'') 


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
  standardize=custom_standardization,
  max_tokens=max_features,
  output_mode='int',
  output_sequence_length=sequence_length)


train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorise_text(text, label): 
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]


train_ds = raw_train_ds.map(vectorise_text)
val_ds = raw_val_ds.map(vectorise_text)
test_ds = raw_test_ds.map(vectorise_text)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
  optimizer='adam',
  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs)

loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)


history_dict = history.history
history_dict.keys()
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()