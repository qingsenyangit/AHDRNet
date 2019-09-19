import os
import torch

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def model_load(model, trained_model_dir, model_file_name):
    model_path = os.path.join(trained_model_dir, model_file_name)
    # trained_model_dir + model_file_name    # '/modelParas.pkl'
    model.load_state_dict(torch.load(model_path))
    return model
