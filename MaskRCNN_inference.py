from mrcnn.config import Config

class InferenceConfig(Config):
    NAME = "FruityConfig"
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.6

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + apples

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128)  # (32, 64, 128, 256, 512) #GYA flower

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3  # GYA lower this number to reduce overlapped detected object
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.3  # GYA higher to generate more proposal?? to fit the the diagonal one?
    RPN_ANCHOR_RATIOS = [0.4, 1.4, 2.4] # GYA
    IMAGE_MIN_DIM = 256
    IMAGE_RESIZE_MODE = 'pad64'


class AppleConfig(InferenceConfig):
    # NAME = "FruityConfig"
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 1
    # IMAGES_PER_GPU = 1
    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.75

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + apples

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # (32, 64, 128, 256, 512) #GYA flower

    # Non-maximum suppression threshold for detection
    # DETECTION_NMS_THRESHOLD = 0.1  # GYA lower this number to reduce overlapped detected object
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # RPN_NMS_THRESHOLD = 0.1
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]  # GYA


class InflorescenceConfig(InferenceConfig):
    # NAME = "FruityConfig"
    # # Set batch size to 1 since we'll be running inference on
    # # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 1
    # IMAGES_PER_GPU = 1
    # # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + apples

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)  # (32, 64, 128, 256, 512) #GYA flower

    # Non-maximum suppression threshold for detection
    # DETECTION_NMS_THRESHOLD = 0.1  # GYA lower this number to reduce overlapped detected object
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # RPN_NMS_THRESHOLD = 0.1
    RPN_ANCHOR_RATIOS = [0.8, 1.6, 2.4]  # GYA


class GrapesConfig(InferenceConfig):
    # NAME = "FruityConfig"
    # # Set batch size to 1 since we'll be running inference on
    # # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 1
    # IMAGES_PER_GPU = 1
    # # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.75

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + apples

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)  # (32, 64, 128, 256, 512) #GYA flower

    # Non-maximum suppression threshold for detection
    # DETECTION_NMS_THRESHOLD = 0.1  # GYA lower this number to reduce overlapped detected object
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # RPN_NMS_THRESHOLD = 0.1
    RPN_ANCHOR_RATIOS = [0.5, 1.3, 2.5]  # GYA
