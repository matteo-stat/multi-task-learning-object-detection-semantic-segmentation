import tensorflow as tf
tf.keras.saving.get_custom_objects().clear()
import ssdseglib


mobilenetv2_builder = ssdseglib.models.MobileNetV2Builder(
    input_image_shape=(480, 640, 3),
    number_of_boxes_per_point=5,
    number_of_classes=4
)
model = mobilenetv2_builder.get_model_for_training()

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
model2 = tf.keras.models.load_model('data/models/mymodel-test.keras', compile=True)

s = 1