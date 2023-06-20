class BaseConfig:
    NUM_TRAIN_EPOCHS: int = 20
    """
    Number of epochs to train for.
    """
    BATCH_SIZE = 64
    """
    Batch size for training.
    - Higher batch sizes can speed up training.
    - Higher batch sizes will use more GPU memory.
    """
    TRAIN_SIZE = 0.8
    EVAL_SIZE = 0.2
    # # Uncomment if you want to downsample
    # TRAIN_SIZE = BATCH_SIZE * 20
    # EVAL_SIZE = int(0.1 * TRAIN_SIZE)

    MAX_INPUT_LENGTH = 256  # max 1024, < 1% of data is > 200 tokens
    """
    Maximum number of tokens to be passed into model input.
    - Higher values may result in better MLM performance (more context).
    - Higher values will use more GPU memory.
    - Refer to `model.config.max_position_embeddings` attribute.
    """
    FP16 = True
    """
    Whether to use 16-bit mixed-precision instead of 32-bit.
    - 16-bit trains faster.
    - Might result in slightly lower MLM accuracy.
    """
    NUM_CPU_WORKERS = 8
    """
    Number of CPU workers for data preprocessing.
    - Speeds up data preprocessing.
    """
    LOGGING_STEPS = 0.5
    """
    How often to log training progress.
    - Proportionate to 1 epoch
    """
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 1e-2
    PUSH_TO_HUB = False
    HUB_MODEL_ID = "username/repo_name"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MLMConfig(BaseConfig):
    MLM_PROB = 0.15


# class TextClassificationConfig(BaseConfig):
#     NUM_LABELS = 2
#     NUM_EPOCHS = 3
#     LEARNING_RATE = 1e-5
#     WARMUP_STEPS = 100
#     WEIGHT_DECAY = 0.01
#     MAX_GRAD_NORM = 1.0
#     LOGGING_STEPS = 100
#     SAVE_STEPS = 100
#     SAVE_TOTAL_LIMIT = 2
#     MODEL_PATH = "bert-base-uncased"
#     MODEL_NAME = "bert-base-uncased"
#     MODEL_SAVE_PATH = "model_save"
