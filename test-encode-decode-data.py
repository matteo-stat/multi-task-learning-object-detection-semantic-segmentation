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
    boxes_scales=(0.1, 0.5),
    additional_square_box=True,  
)
boxes_default.calculate_boxes_coordinates(image_shape=INPUT_IMAGE_SHAPE)

# create a data reader encoder
data_reader_encoder = ssdseglib.datacoder.DataEncoderDecoder(
    num_classes=4,
    image_shape=INPUT_IMAGE_SHAPE,
    xmin_boxes_default=boxes_default.xmin,
    ymin_boxes_default=boxes_default.ymin,
    xmax_boxes_default=boxes_default.xmax,
    ymax_boxes_default=boxes_default.ymax,
    iou_threshold=0.5,
    std_offsets=(0.1, 0.1, 0.2, 0.2),
    augmentation_horizontal_flip=True
)

# load metadata
with open('data/train.json', 'r') as f:
    path_images_train, path_masks_train, path_labels_boxes_train = map(list, zip(*json.load(f)))

# read and encode data
image, mask, labels_boxes = data_reader_encoder.read_and_encode(path_images_train[0], path_masks_train[0], path_labels_boxes_train[0])

# decode data
# this should return the original bounding boxes coordinates if there were default bounding boxes overlapped
decoded = data_reader_encoder.decode_to_corners(labels_boxes[:, -4], labels_boxes[:, -3], labels_boxes[:, -2], labels_boxes[:, -1])

# keep only decoded default bounding boxes that were not background
decoded_not_background = tf.boolean_mask(
    tensor=tf.stack(decoded, axis=1),
    mask=tf.math.greater(tf.math.reduce_sum(labels_boxes[:, :4], axis=1), 0.0),
    axis=0
)
