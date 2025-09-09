# EZ-VC: Easy Zero-shot Any-to-Any Voice Conversion

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/EZ-VC/EZ-VC)
[![arXiv](https://img.shields.io/badge/arXiv-2505.16691-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.16691)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://ez-vc.github.io/)
[![lab](https://img.shields.io/badge/SPRING-Lab-grey?labelColor=lightgrey)](https://asr.iitm.ac.in/)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->


### Our paper has been accepted to the Findings of EMNLP 2025!

## Installation

### Create a separate environment if needed

```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n ez-vc python=3.10
conda activate ez-vc
```

### Install PyTorch with matched device

<details>
<summary>NVIDIA GPU</summary>

> ```bash
> # Install pytorch with your CUDA version, e.g.
> pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
> ```

</details>

<details>
<summary>AMD GPU</summary>

> ```bash
> # Install pytorch with your ROCm version (Linux only), e.g.
> pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
> ```

</details>

<details>
<summary>Intel GPU</summary>

> ```bash
> # Install pytorch with your XPU version, e.g.
> # Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit must be installed
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/test/xpu
> 
> # Intel GPU support is also available through IPEX (Intel® Extension for PyTorch)
> # IPEX does not require the Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit
> # See: https://pytorch-extension.intel.com/installation?request=platform
> ```

</details>

<details>
<summary>Apple Silicon</summary>

> ```bash
> # Install the stable pytorch, e.g.
> pip install torch torchaudio
> ```

</details>

### Then follow instructions below:


### Local installation

```bash
git clone https://github.com/EZ-VC/EZ-VC
cd EZ-VC
git submodule update --init --recursive
pip install -e .

# Install espnet for xeus (Exactly this version)
pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl'
```

## Inference

We have provided a Jupyter notebook for inference.

Open [Inference notebook](src/f5_tts/infer/infer.ipynb).

Run all. 

The converted audio will be available at the last cell.


## Acknowledgements

- [F5-TTS](https://arxiv.org/abs/2410.06885) for opensourcing their code which has made EZ-VC possible.

## Citation
If our work and codebase is useful for you, please cite as:
```
@misc{joglekar2025ezvceasyzeroshotanytoany,
      title={EZ-VC: Easy Zero-shot Any-to-Any Voice Conversion}, 
      author={Advait Joglekar and Divyanshu Singh and Rooshil Rohit Bhatia and S. Umesh},
      year={2025},
      eprint={2505.16691},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2505.16691}, 
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license. Sorry for any inconvenience this may cause.
