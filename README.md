# ANMAF: Automated Neuronal Morphology Analysis Framework using Convolutional Neural Networks

ANMAF is an automated framework for detecting cells in neuronal images using [Mask R-CNN](https://github.com/matterport/Mask_RCNN). It is designed for the purpose of facilitating neuronal morphology analysis by enabling fast and less-cumbersome way of collecting neuronal morphology data. Details can be found in [Tong et al. (2021)](https://www.nature.com/articles/s41598-021-87471-w).

## Installation
The easiest way of reproducing our work should be by using [conda](https://conda.io/). Assuming that you have Anaconda installed and configured on your system, run the following commands in the command prompt/terminal to create a virtual environment and activate it.
```bash
conda create -n anmaf python=3.8
conda activate anmaf
```

Once the environment is created and activated, navigate into the directory where you cloned this repository. Then run the following command to install all the required 3rd-party libraries.
```bash
pip install -r requirements.txt
```

Additionally, you would need [TensorFlow](https://www.tensorflow.org/) installed in your environment. Assuming [CUDA](https://developer.nvidia.com/cuda-toolkit) and [CuDNN](https://developer.nvidia.com/cudnn) are already configured correctly, TensorFlow can be installed via the following command:
```bash
pip install tensorflow
```
See the [official TensorFlow documentation](https://www.tensorflow.org/install/gpu) for more details.


## Getting Started

```bash
python ANMAF.py
```

TODO: run ANMAF with custom arguments


## Training ANMAF with Custom Data

### Data Generation
The first step to train ANMAF with custom data is to generate a synthetic dataset. For more details, see the [synthetic data generation guide](data/README.md).

### Training
TODO: Custom training guide.

## Publication
ANMAF is free of use for both commercial and non-commercial purposes. If you decide to use ANMAF, please cite the following article.

```
@article{Tong2021ANMAF,
  author={Ling Tong and Rachel Langton and Joseph Glykys and Stephen Baek},
  title={{ANMAF}: an automated neuronal morphology analysis framework using convolutional neural networks},
  journal={Scientific Reports},
  volume={11},
  pages={Article No. 8179},
  month = {4},
  year{2021},
  doi={https://doi.org/10.1038/s41598-021-87471-w},
}
```

## Acknowledgement
This work was funded by the National Institutes of Health (NIH) - National Institute of Neurological Disorders and Stroke (NINDS) through the Grant No. 1R01NS115800 (PI: Joseph Glykys, University of Iowa) and partially through the Grant No. K08NS091248 (PI: Joseph Glykys). The Iowa Neuroscience Institute also sponsored this work.

## Contributors

- [Ling Tong](https://tippie.uiowa.edu/people/ling-tong), University of Iowa
- [Stephen Baek](http://www.stephenbaek.com), University of Virginia
- [Joseph Glykys](https://medicine.uiowa.edu/pediatrics/profile/joseph-glykys), University of Iowa
