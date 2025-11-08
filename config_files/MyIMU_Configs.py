"""
Configuration file for your IMU dataset (10 classes).

INSTRUCTIONS:
1. Adjust input_channels based on your sensor data (e.g., 6 for acc+gyro)
2. Adjust features_len if needed (depends on your model architecture)
3. Keep num_classes = 10
4. Adjust num_epoch, batch_size based on your dataset size

After creating your dataset with prepare_csv_dataset.py, rename this file to match
your dataset name (e.g., if output_dir was data/ActivityIMU, rename to ActivityIMU_Configs.py)
"""


class Config(object):
    def __init__(self):
        # Model configs
        self.input_channels = 6  # CHANGE: number of sensor channels (e.g., 6 for 3-axis acc + gyro)
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 10  # Your 10 activity classes
        self.dropout = 0.35
        self.features_len = 18  # Can adjust based on sequence length

        # Training configs
        self.num_epoch = 40  # Increase if needed (e.g., 60-80 for better results)

        # Optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # Data parameters
        self.drop_last = True
        self.batch_size = 128  # Reduce if GPU memory issues (e.g., 64, 32)

        # Contrastive learning configs
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        """
        Data augmentation for contrastive learning.
        These augmentations create different views of the same sample.
        """
        self.jitter_scale_ratio = 1.1  # Scaling augmentation
        self.jitter_ratio = 0.8        # Jittering strength
        self.max_seg = 8               # Max segments for permutation augmentation


class Context_Cont_configs(object):
    def __init__(self):
        """
        Contextual contrastive learning parameters.
        """
        self.temperature = 0.2          # Temperature for contrastive loss
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        """
        Temporal contrastive learning parameters.
        """
        self.hidden_dim = 100
        self.timesteps = 6
