import torch

MAIN_SET = {
    'characters': "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'OUTPUT_PATH': "./saved_models/",
}

TRAIN_SET = {
    'image_size': 160 * 80,
    'num_epoc': 1,
}

DATA_SET = {
    'TRAIN_NUM': 20000,
    'TEST_NUM': 2000,
    'TRAIN_PATH': "./dataset/train_dataset/",
    'TRAIN_FILE': './dataset/train_label.txt',
    "TEST_PATH": "./dataset/test_dataset/",
    "TEST_FILE": "./dataset/test_label.txt",
}
