# ============================================================
# MODEL / ARCHITECTURE
# ============================================================

LEVELS = (
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
)

# Defined by the paper.
STRIDES = {
    "P3": 8,
    "P4": 16,
    "P5": 32,
    "P6": 64,
    "P7": 128,
}

# Current project choice.
IMAGE_SIZE = 512

INPUT_CHANNELS = 3


# ============================================================
# DATASET
# ============================================================

CSV_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_labels.csv"
)

TRAIN_DCM_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_images"
)

VAL_RATIO = 0.2

SEED = 42


# ============================================================
# DATALOADER
# ============================================================

BATCH_SIZE = 8

TRAIN_NUM_WORKERS = 2
VAL_NUM_WORKERS = 2

TRAIN_SHUFFLE = True

VAL_SHUFFLE = False


# ============================================================
# OPTIMIZATION
# ============================================================

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.0

NUM_EPOCHS = 10


# ============================================================
# SCHEDULER
# ============================================================

USE_SCHEDULER = True

LR_STEP_SIZE = 100

LR_GAMMA = 0.1


# ============================================================
# LOSS
# ============================================================

CENTER_LOSS_WEIGHT = 1.0

REGRESSION_LOSS_WEIGHT = 1.0

CENTERNESS_LOSS_WEIGHT = 1.0


# ============================================================
# INFERENCE / POST-PROCESSING
# ============================================================

SCORE_THRESHOLD = 0.1

NMS_THRESHOLD = 0.5


# ============================================================
# EVALUATION
# ============================================================

EVAL_IOU_THRESHOLD = 0.5

AR_MAX_DETECTIONS = 10


# ============================================================
# LOGGING
# ============================================================

LOG_SCALARS = True

LOG_HISTOGRAMS = False

LOG_GRADIENTS = False

LOG_HPARAMS = False

HISTOGRAM_EVERY_N_EPOCHS = 5

GRADIENT_EVERY_N_STEPS = 100


# ============================================================
# EXPERIMENTS / CHECKPOINTS
# ============================================================

# Existing root directory.
# The training code will create this directory if necessary.
EXPERIMENTS_DIR = "checkpoints"

# Existing pretrained ResNet-50 backbone.
# This is NOT a detector checkpoint.
RESNET50_CHEST_XRAY_CHECKPOINT = (
    "checkpoints/"
    "resnet50_scratch_chest_xray_best.pth"
)

LAST_CHECKPOINT_NAME = "last.pt"

BEST_CHECKPOINT_NAME = "best.pt"

SAVE_EVERY_N_EPOCHS = 1


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda"


# ============================================================
# MIXED PRECISION
# ============================================================

USE_AMP = False


# ============================================================
# GRADIENTS
# ============================================================

GRADIENT_CLIP_NORM = None