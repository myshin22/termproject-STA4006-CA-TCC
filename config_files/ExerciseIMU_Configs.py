"""
Configuration file for Exercise IMU dataset (10 classes).

Dataset: Wearable IMU sensors (left + right wrist)
- 12 channels: 6 per wrist (3-axis acc + 3-axis gyro)
- 10 exercise classes
- Window: 5 seconds (330 frames @ 66Hz)
"""


class Config(object):
    def __init__(self):
        # Model configs
        self.input_channels = 12  # Left + Right: 3-axis acc + 3-axis gyro each
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 10  # 10 exercise types
        self.dropout = 0.35
        self.features_len = 43  # For sequence length 330: (330-8)/1+1 = 323, after pooling ~43

        # Training configs
        self.num_epoch = 40

        # Optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # Data parameters
        self.drop_last = True
        self.batch_size = 128

        # Contrastive learning configs
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        """
        Data augmentation for contrastive learning.
        """
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        """
        Contextual contrastive learning parameters.
        """
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        """
        Temporal contrastive learning parameters.
        """
        self.hidden_dim = 100
        self.timesteps = 6
