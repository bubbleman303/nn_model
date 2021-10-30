BASE_DIR = __file__.replace("src/conf/config.py", "").replace("src\conf\config.py", "")
TRAIN_DATA_DIR = f"{BASE_DIR}train_data/" + "{}"
TRAIN_IMAGE_DIR = BASE_DIR + "image/train_image_set/{}"
NN_PARAM_DIR = BASE_DIR + "nn_params/{}"
