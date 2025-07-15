import os

class Config:
    # Data paths
    BASE_DATA_PATH = './base_data_path'
    DATA_SPLITS_PATH = os.path.join(BASE_DATA_PATH, './data_splits_path')
    FULL_DATA_PATH = os.path.join(BASE_DATA_PATH, './full_data_path')
    SAVE_MODEL_PATH = os.path.join(BASE_DATA_PATH, './save_model_path')
    
    # Image parameters
    IMAGE_SIZE = (128, 128)
    NUM_CHANNELS = 3  # RGB channels for ResNet
    PERCENTAGE_CLIP = 99
    ZERO_CENTERED = False
    
    # Dataset parameters
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    
    # Model parameters
    PRETRAINED = True
    HIDDEN_SIZE = 256
    DROPOUT_RATE = 0.3
    
    # Training parameters
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MODE = 'min'
    
    # File names
    BEST_MODEL_NAME = 'best_model.pth'
    TRAINING_HISTORY_NAME = 'training_history.csv'
    RESULTS_NAME = 'results.csv'
    KNEE_DATA_FILE = 'knee.npz'
    
    # Training data files
    TRAIN_DATA_FILE = 'train_data.csv'
    VAL_DATA_FILE = 'val_data.csv'
    TEST_DATA_FILE = 'test_data.csv'
    
    @classmethod
    def get_data_file_path(cls, phase):
        """Get the path to the data file for a given phase"""
        filename = f'{phase}_data.csv'
        return os.path.join(cls.DATA_SPLITS_PATH, filename)
    
    @classmethod
    def get_knee_data_path(cls):
        """Get the path to the knee data file"""
        return os.path.join(cls.FULL_DATA_PATH, cls.KNEE_DATA_FILE)
    
    @classmethod
    def get_save_path(cls, filename):
        """Get the full save path for a given filename"""
        return os.path.join(cls.SAVE_MODEL_PATH, filename)
    
    @classmethod
    def ensure_save_dir(cls):
        """Ensure the save directory exists"""
        os.makedirs(cls.SAVE_MODEL_PATH, exist_ok=True) 