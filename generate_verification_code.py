import shutil
from captcha.image import ImageCaptcha
from config import DATA_SET
import random
import string
import os


def generate_train_dataset():
    output_dir = DATA_SET["TRAIN_PATH"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = ImageCaptcha(width=160, height=80)
    characters = string.ascii_uppercase + string.digits

    with open(DATA_SET["TRAIN_FILE"], "w") as f:
        for i in range(DATA_SET["TRAIN_NUM"]):
            captcha_text = ''.join(random.choices(characters, k=5))
            image_path = os.path.join(output_dir, f"{i:05}.png")
            image.write(captcha_text, image_path)
            f.write(f"{captcha_text}\n")


def generate_test_dataset():
    output_dir = DATA_SET["TEST_PATH"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = ImageCaptcha(width=160, height=80)
    characters = string.ascii_uppercase + string.digits

    with open(DATA_SET["TEST_FILE"], "w") as f:
        for i in range(DATA_SET["TEST_NUM"]):
            captcha_text = ''.join(random.choices(characters, k=5))
            image_path = os.path.join(output_dir, f"{i:05}.png")
            image.write(captcha_text, image_path)
            f.write(f"{captcha_text}\n")


def remove_train_dataset():
    output_dir = DATA_SET["TRAIN_PATH"]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    with open(DATA_SET["TRAIN_FILE"], "w") as f:
        pass


def remove_test_dataset():
    output_dir = DATA_SET["TEST_PATH"]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    with open(DATA_SET["TEST_FILE"], "w") as f:
        pass


mode = eval(input())
if mode == 1:
    generate_train_dataset()
    print("Train dataset generated")
elif mode == 2:
    generate_test_dataset()
    print("Test dataset generated")
elif mode == 3:
    remove_train_dataset()
    print("Train dataset removed")
elif mode == 4:
    remove_test_dataset()
    print("Test dataset removed")
else:
    print("Invalid mode")
