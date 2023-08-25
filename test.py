import tensorflow as tf
tf.keras.saving.get_custom_objects().clear()
import ssdseglib

# global variables
INPUT_IMAGE_SHAPE = (480, 640, 3)
STANDARD_DEVIATIONS_CENTROIDS_OFFSETS = (0.1, 0.1, 0.2, 0.2)
NUMBER_OF_CLASSES = 4
SHUFFLE_BUFFER_SIZE = 512
BATCH_SIZE = 16
SEED = 1993

# create default bounding boxes
boxes_default = ssdseglib.boxes.DefaultBoundingBoxes(
    feature_maps_shapes=((30, 40), (15, 20), (8, 10), (4, 5)),
    centers_padding_from_borders_percentage=0.025,
    boxes_scales=(0.2, 0.9),
    additional_square_box=True,  
)

# scale default bounding boxes to image shape
boxes_default.rescale_boxes_coordinates(image_shape=INPUT_IMAGE_SHAPE[:2])

# create a data reader encoder
data_reader_encoder = ssdseglib.datacoder.DataEncoderDecoder(
    num_classes=4,
    image_shape=INPUT_IMAGE_SHAPE[:2],
    xmin_boxes_default=boxes_default.get_boxes_coordinates_xmin(coordinates_style='ssd'),
    ymin_boxes_default=boxes_default.get_boxes_coordinates_ymin(coordinates_style='ssd'),
    xmax_boxes_default=boxes_default.get_boxes_coordinates_xmax(coordinates_style='ssd'),
    ymax_boxes_default=boxes_default.get_boxes_coordinates_ymax(coordinates_style='ssd'),
    iou_threshold=0.5,
    standard_deviations_centroids_offsets=STANDARD_DEVIATIONS_CENTROIDS_OFFSETS,
    augmentation_horizontal_flip=True
)

# model builder
model_builder = ssdseglib.models.MobileNetV2SsdSegBuilder(
    input_image_shape=INPUT_IMAGE_SHAPE,
    number_of_boxes_per_point=[
        len(aspect_ratios) + (1 if boxes_default.additional_square_box else 0)
        for aspect_ratios in boxes_default.feature_maps_aspect_ratios
    ],
    number_of_classes=NUMBER_OF_CLASSES,
    center_x_boxes_default=boxes_default.get_boxes_coordinates_center_x(coordinates_style='ssd'),
    center_y_boxes_default=boxes_default.get_boxes_coordinates_center_y(coordinates_style='ssd'),
    width_boxes_default=boxes_default.get_boxes_coordinates_width(coordinates_style='ssd'),
    height_boxes_default=boxes_default.get_boxes_coordinates_height(coordinates_style='ssd'),
    standard_deviations_centroids_offsets=STANDARD_DEVIATIONS_CENTROIDS_OFFSETS
)

# model for training
model = model_builder.get_model_for_training()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

dice_loss = ssdseglib.losses.dice(classes_weights=(0.175, 0.275, 0.275, 0.275))

# weighted metrics for semantic segmentation
jaccard_iou_segmentation_masks_metric = ssdseglib.metrics.jaccard_iou_segmentation_masks(classes_weights=(0.175, 0.275, 0.275, 0.275))

# weighted metrics for boxes classification
categorical_accuracy_metric = ssdseglib.metrics.categorical_accuracy(classes_weights=(0.175, 0.275, 0.275, 0.275))


# each ouput has its own loss and metrics
model.compile(
    optimizer=optimizer,
    loss={
        'output-mask': dice_loss,
        'output-labels': ssdseglib.losses.confidence_loss,
        'output-boxes': ssdseglib.losses.localization_loss
    },
    loss_weights={
        'output-mask': 1.0,
        'output-labels': 1.0,
        'output-boxes': 1.0
    },
    metrics={
        'output-mask': jaccard_iou_segmentation_masks_metric,
        'output-labels': categorical_accuracy_metric,
    }
)

model.save('data/models/mymodel-test.keras')

s = 1

# load model
model_trained = tf.keras.models.load_model('data/models/mymodel-test.keras', compile=True)

s = 1

# transfer weights
model_inference = model_builder.get_model_for_inference(
    model_trained=model_trained,
    max_number_of_boxes_per_class=6,
    max_number_of_boxes_per_sample=20,
    boxes_iou_threshold=0.5,
    labels_probability_threshold=0.6
)

s = 2
