import os
import shutil


def wandb_store_file(wandb_run, file_path, target_file_name=None):
    if target_file_name is None:
        target_file_name = os.path.basename(file_path)

    wandb_path = os.path.join(wandb_run.dir, target_file_name)
    if os.path.exists(wandb_path):
        os.remove(wandb_path)

    shutil.copyfile(file_path, wandb_path)
    wandb_run.save(wandb_path, os.path.dirname(wandb_path))
