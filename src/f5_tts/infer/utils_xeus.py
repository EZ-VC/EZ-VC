import json
import joblib
import torch
import librosa
from cached_path import cached_path
from espnet2.tasks.ssl import SSLTask

xeus_path = str(cached_path(f"hf://espnet/xeus/model/xeus_checkpoint_old.pth"))
km_path = str(cached_path(f"hf://SPRINGLab/EZ-VC/kmeans_xeus_500_multilingual.pkl"))
km_model = joblib.load(km_path)
unit_map = json.load(open("xeus/char_map.json"))

# device = "cuda" if torch.cuda.is_available() else "cpu"

class ApplyKmeans:
    def __init__(self, device):
        self.km_model = km_model
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.device = device

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    def __call__(self, x):
        x = x.to(self.device)
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(x, self.C)
            + self.Cnorm
        )
        return dist.argmin(dim=1).cpu().numpy()

# apply_kmeans = ApplyKmeans(device)

# Load XEUS model from checkpoint
def load_xeus_model(device):
    xeus_model, _ = SSLTask.build_model_from_file(
        'xeus/config.yaml',
        xeus_path,
        device,
        )
    return xeus_model

def deduplicate_units(units):
    if not units:
        return ""
    deduplicated_sequence = [units[0]]
    for unit in units[1:]:
        if unit != deduplicated_sequence[-1]:
            deduplicated_sequence.append(unit)
    # Convert to string
    deduplicated_sequence = ''.join([str(unit) for unit in deduplicated_sequence])
    return deduplicated_sequence

def extract_units(audio_path, xeus_model, apply_kmeans, device):

    audio_array, _ =  librosa.load(audio_path, sr=16000)

    # Convert to tensor
    wav_lengths = torch.LongTensor([len(audio_array)]).to(device)
    wav_tensor = torch.Tensor([audio_array]).to(device)

    with torch.no_grad():
            # Encode and get hidden states
            outputs = xeus_model.encode(
                wav_tensor, wav_lengths, use_mask=False, use_final_output=False
            )

    features = outputs[0][14].squeeze()
    # features_tensor = torch.from_numpy(features).cuda()
    
    units = apply_kmeans(features).tolist()
    # print(units)
    unit_string = "".join([unit_map[str(unit)] for unit in units])
    units = deduplicate_units(unit_string)
    return units