import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Reshape
from tensorflow.keras.models import Model

# Create lists for generating data
my_data = list(range(100))

# Define a function to load and preprocess the data from file paths
def load_and_preprocess_data(useless_input):
    x = tf.random.uniform(shape=(10, 10, 3), minval=0, maxval=10)
    y_mask = tf.random.uniform(shape=(10, 10, 3), minval=0, maxval=10)
    y_encoded_boxes = tf.random.uniform(shape=(100, 3), minval=0, maxval=100)
    
    return x, {'output_mask': y_mask, 'output_encoded_boxes': y_encoded_boxes}

# tensorflow train dataset pipeline
ds_train = (
    tf.data.Dataset.from_tensor_slices(my_data)
    .map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=6)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# check if generator works fine
# for input_batch, target_batch in ds_train.take(10):
#     for image_sample, mask_sample, labels_boxes_sample in zip(input_batch, target_batch['output_mask'], target_batch['output_encoded_boxes']):
#         s=0


# Define your custom loss functions
def loss_mask(y_true, y_pred):
    # Define your custom loss calculation for y_mask
    # This could be something like binary cross-entropy
    return tf.reduce_mean((y_true))

def loss_encoded_boxes(y_true, y_pred):
    # Define your custom loss calculation for y_encoded_boxes
    # This could be something like mean squared error
    return tf.reduce_mean((y_pred))

# Create the model
def create_model():
    # inputs
    inputs = Input(shape=(10, 10, 3))
    
    # backbone
    shared_layer = Conv2D(3, (3,3), padding='same', activation='relu')(inputs)
    
    # output segmentation
    output_mask = Conv2D(3, (3,3), padding='same', activation='relu', name='output_mask')(shared_layer)

    # output detection
    output_encoded_boxes = Conv2D(3, (3,3), padding='same', activation='relu')(shared_layer)
    output_encoded_boxes_reshaped = Reshape((100, 3), name='output_encoded_boxes')(output_encoded_boxes)
    
    model = Model(inputs=inputs, outputs=[output_mask, output_encoded_boxes_reshaped])
    
    return model

# Create the model
model = create_model()

# Compile the model with different loss functions for each output
model.compile(optimizer='adam',
              loss={'output_mask': loss_mask, 'output_encoded_boxes': loss_encoded_boxes},
              loss_weights={'output_mask': 1.0, 'output_encoded_boxes': 1.0})

# # Train the model using the dataset
num_epochs = 10
model.fit(ds_train, epochs=num_epochs)