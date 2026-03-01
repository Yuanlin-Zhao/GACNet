# 🛠️ Installation

### 1. Create Environment
We recommend using Conda to manage your dependencies:
```bash
conda create -n gcanet python=3.9 -y
conda activate gcanet
```

### 2. Install Dependencies
Install the required packages via pip:
```bash
pip install -r requirements.txt
```
It is recommended to use GPU for training and testing.
---

## 📂 Dataset Preparation
Evaluate the model on the **RGBTDrone** dataset for download.

* **Download Link:** [Baidu Netdisk](https://pan.baidu.com/s/1WVWRA3ALyzsJd0kSTDWWOA?pwd=dmuy)
* **Extraction Code:** `dmuy`

### 📥 权重下载 (Weight Download)


| Model Variant | Dataset | Download Link | Extraction Code |
| :--- | :--- | :--- | :--- |
| **GACNet-T (RGBDrone)** | RGBDrone | [Baidu Netdisk](https://pan.baidu.com/s/1AzvDIDhJtk3CJVpPTGRkZA) | `161n` |
| **GACNet-S (RGBDrone)** | RGBDrone | [Baidu Netdisk](https://pan.baidu.com/s/181NOh17OYWIKI6P_Gxf4XA?pwd) | `qkbe` |
| **GACNet-L (RGBDrone)** | RGBDrone | [Baidu Netdisk](https://pan.baidu.com/s/1rG2q2WfblPl9RHfRqyB5Vg) | `1mdh` |

**Directory Structure:**
After downloading and unzipping, please organize the data as follows:
```text
datasets/
└── RGBTDronePerson/
    ├── train/
    │   ├── hr/          # High-Resolution Ground Truth
    │   └── lr/          # Low-Resolution (x4Thermal)
    └── val/
        ├── hr/
        └── lr/
```

---

## 🚀 Usage

This project is built on the **BasicSR** framework. All configurations are managed via `.yml` files.

### Training
To start training the GCANet model:
```bash
python basicsr/train.py -opt options/train/train_GCANet_x4.yml
```
num_block: 1 GACNet-T, num_block: 6 GACNet-S, num_block: 12 GACNet-L
### Testing
To evaluate the model and calculate metrics (PSNR/SSIM):
```bash
python basicsr/test.py -opt options/test/test_GCANet_x4.yml
```

## 🤝 Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR). We thank the authors for their excellent open-source framework.

---
**Contact:** `zyl@chd.edu.cn`
