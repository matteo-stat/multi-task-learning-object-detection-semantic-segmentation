import ssdseglib
import json
import tensorflow as tf

# check gpu and set an higher memory limit
gpus = tf.config.list_physical_devices('GPU')
if gpus: 
    tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=7400)])
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

# global variables
INPUT_IMAGE_SHAPE = (480, 640)
SHUFFLE_BUFFER_SIZE = 512
BATCH_SIZE = 8
SEED = 1993

# create default bounding boxes
boxes_default = ssdseglib.boxes.DefaultBoundingBoxes(
    feature_maps_shapes=((30, 40), (15, 20), (8, 10), (4, 5)),
    centers_padding_from_borders_percentage=0.025,
    boxes_scales=(0.2, 0.9),
    additional_square_box=True,  
)

# rescale default bounding boxes to input image shape
boxes_default.rescale_boxes_coordinates(image_shape=INPUT_IMAGE_SHAPE)

# create a data reader encoder
data_reader_encoder = ssdseglib.datacoder.DataEncoderDecoder(
    num_classes=4,
    image_shape=INPUT_IMAGE_SHAPE,
    xmin_boxes_default=boxes_default.get_boxes_coordinates_xmin(coordinates_style='ssd'),
    ymin_boxes_default=boxes_default.get_boxes_coordinates_ymin(coordinates_style='ssd'),
    xmax_boxes_default=boxes_default.get_boxes_coordinates_xmax(coordinates_style='ssd'),
    ymax_boxes_default=boxes_default.get_boxes_coordinates_ymax(coordinates_style='ssd'),
    iou_threshold=0.5,
    standard_deviations_centroids_offsets=(0.1, 0.1, 0.2, 0.2),
    augmentation_horizontal_flip=True
)

# load metadata
with open('data/train.json', 'r') as f:
    path_images_train, path_masks_train, path_labels_boxes_train = map(list, zip(*json.load(f)))

# tensorflow train dataset pipeline
ds_train = (
    tf.data.Dataset.from_tensor_slices((path_images_train, path_masks_train, path_labels_boxes_train))
    .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    .map(data_reader_encoder.read_and_encode, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    .map(ssdseglib.datacoder.augmentation_rgb_channels, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

model = ssdseglib.models.build_mobilenetv2_ssdseg(number_of_boxes_per_point=6, number_of_classes=4)

# Compile the model with different loss functions for each output
model.compile(
    optimizer='adam',
    loss={
        'output-mask': tf.keras.losses.categorical_crossentropy,
        'output-labels': ssdseglib.losses.confidence_loss,
        'output-boxes': ssdseglib.losses.localization_loss
    },
    loss_weights={
        'output-mask': 1.0,
        'output-labels': 1.0,
        'output-boxes': 1.0
    },
    metrics={
        'output-mask': tf.keras.metrics.CategoricalAccuracy()
    }
)

# Train the model using the dataset
num_epochs = 2
model.fit(ds_train, epochs=num_epochs)