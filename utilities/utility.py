import json
import os


def get_model_weights(model_name):
    json_file_path = "model_weights.json"

    # Read the JSON file
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    # Access model paths
    model_weights_path = data["models"][model_name]

    return model_weights_path


def prone_static_dir(folder_path):
    try:
        # Get the list of files and directories in the folder
        contents = os.listdir(folder_path)

        # Iterate over the contents and remove each one
        for item in contents:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                os.rmdir(item_path)

        print("Contents of '{}' deleted successfully.".format(folder_path))
    except Exception as e:
        print(f"An error occurred: {e}")



