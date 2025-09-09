import torch
from safetensors.torch import save_file
def extract_and_save_ema_model(checkpoint_path, safetensors):
    try:
        checkpoint = torch.load(checkpoint_path)
        print("Original Checkpoint Keys:", checkpoint.keys())

        ema_model_state_dict = checkpoint.get("ema_model_state_dict", None)
        if ema_model_state_dict is None:
            return "No 'ema_model_state_dict' found in the checkpoint."

        if safetensors:
            checkpoint_path = checkpoint_path.replace(".pt", ".safetensors")
            save_file(ema_model_state_dict, checkpoint_path)

        print(f"New checkpoint saved at: {checkpoint_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

chk='model_2700000.pt'
extract_and_save_ema_model(chk, True)
