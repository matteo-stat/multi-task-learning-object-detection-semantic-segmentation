import ssdseglib
import json
import tensorflow as tf

# global variables
INPUT_IMAGE_SHAPE = (480, 640)
SHUFFLE_BUFFER_SIZE = 512
BATCH_SIZE = 16
SEED = 1993

# create default bounding boxes
boxes_default = ssdseglib.boxes.DefaultBoundingBoxes(
    feature_maps_shapes=((24, 32), (12, 16), (6, 8), (3, 4)),
    feature_maps_aspect_ratios=(1.0, 2.0, 3.0, 1/2, 1/3),
    centers_padding_from_borders=0.5,
    boxes_scales=(0.2, 0.9),
    additional_square_box=True,  
)
boxes_default.calculate_boxes_coordinates(image_shape=INPUT_IMAGE_SHAPE)

# create a data reader encoder
data_reader_encoder = ssdseglib.datacoder.DataReaderEncoder(
    num_classes=4,
    xmin_boxes_default=boxes_default.xmin,
    ymin_boxes_default=boxes_default.ymin,
    xmax_boxes_default=boxes_default.xmax,
    ymax_boxes_default=boxes_default.ymax,
    iou_threshold=0.5,
    offsets_std=(0.1, 0.1, 0.2, 0.2),
)

# load metadata
with open('data/train.json', 'r') as f:
    path_images_train, path_masks_train, path_labels_boxes_train = map(list, zip(*json.load(f)))

# simple check to test the data reader encoder
# data_reader_encoder.read_encode(path_images_train[0], path_masks_train[0], path_labels_boxes_train[0])

# tensorflow train dataset pipeline
ds_train = (
    tf.data.Dataset.from_tensor_slices((path_images_train, path_masks_train, path_labels_boxes_train))
    .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    .map(data_reader_encoder.read_encode, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size=BATCH_SIZE)
    #.map(dataAugmentation)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# check if images are loaded fine
for image_batch, mask_batch, labels_boxes_encoded in ds_train.take(4):
    k = 0

