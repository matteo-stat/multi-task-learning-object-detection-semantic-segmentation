import ssdseglib
import tensorflow as tf

# params
batch_size = 8
num_boxes_per_batch = 8000
num_channels = 4
num_background = int(0.8 * num_boxes_per_batch)

# simulate y_true
y_true_not_background = tf.random.normal(shape=(batch_size, num_boxes_per_batch - num_background, num_channels))
y_true_background = tf.zeros(shape=(batch_size, num_background, num_channels))
y_true = tf.concat([y_true_background, y_true_not_background], axis=1)
y_true = tf.reshape(y_true, (-1, num_channels))
y_true = tf.random.shuffle(y_true)
y_true = tf.reshape(y_true, (batch_size, num_boxes_per_batch, num_channels))

# simulate y_pred
y_pred = tf.random.normal(shape=(batch_size, num_boxes_per_batch, num_channels))

# --------------------------------------------------------------------------------------------------
INPUT_IMAGE_SHAPE = (480, 640)

# create default bounding boxes
boxes_default = ssdseglib.boxes.DefaultBoundingBoxes(
    feature_maps_shapes=((30, 40), (15, 20), (8, 10), (4, 5)),
    centers_padding_from_borders_percentage=0.025,
    additional_square_box=False,  
)
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

jaccard_iou_bounding_boxes = ssdseglib.metrics.jaccard_iou_bounding_boxes(
    center_x_boxes_default=data_reader_encoder.center_x_boxes_default,
    center_y_boxes_default=data_reader_encoder.center_y_boxes_default,
    width_boxes_default=data_reader_encoder.width_boxes_default,
    height_boxes_default=data_reader_encoder.height_boxes_default,
    standard_deviation_center_x_offsets=data_reader_encoder.standard_deviation_center_x_offsets,
    standard_deviation_center_y_offsets=data_reader_encoder.standard_deviation_center_y_offsets,
    standard_deviation_width_offsets=data_reader_encoder.standard_deviation_width_offsets,
    standard_deviation_height_offsets=data_reader_encoder.standard_deviation_height_offsets
)

print(jaccard_iou_bounding_boxes(y_true, y_pred))
