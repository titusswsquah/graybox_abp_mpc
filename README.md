# Physics-informed Neural Model Predictive Control of Interacting Active Brownian Particles
Code for the paper ["Physics-informed Neural Model Predictive Control of Interacting Active Brownian Particles"](https://arxiv.org/abs/xxxx.xxxxx) by Titus Quah, Sho C. Takatori, and James B. Rawlings. 

Website: https://titusswsquah.github.io/graybox_abp_mpc/
<!-- <div align="center">
  <img src="./media/fig1.png" alt="Overview of graybox+mpc" class="image-900x544">
</div> -->
<div align="center">
    <img src="./media/gif3.gif" alt="MPC (active hard disks)" class="image-900x1125">
</div>


## Prerequisites
To install the dependencies, you need to create a conda environment using the `environment.yml` file. Follow these steps:

1. Ensure you have Anaconda or Miniconda installed. If not, download and install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. Navigate to the project directory:
    ```sh
    cd /path/to/graybox_abp_mpc
    ```

3. Create the conda environment:
    ```sh
    conda env create -f environment.yml
    ```

4. Activate the environment:
    ```sh
    conda activate graybox_abp_mpc
    ```

This will install all the necessary dependencies listed in the `environment.yml` file.
There is one more dependency that is not listed in the `environment.yml` file, which is the `mpctools` package. To install this package, clone the repository from [here](https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/) and install it using the following commands:

```sh 
pip install -e.
```
## Running the code
To control the system using the trained model as shown in the video above, run the following command:
```sh
python ctrl.py
```
## Citation
If you find this code useful, please consider citing our paper:
```
@misc{quah2025abpgrayboxmpc,
  title={Physics-informed neural model predictive control of interacting active {Brownian} particles},
  author={Quah, Titus and Takatori, Sho C and Rawlings, James B},
  year={2025}
}
```
<style>
  /* 900x544 image */
  .image-900x544 {
    width: 900px;
    height: 544px;
  }
  /* 900x600 image */
  .image-900x600 {
    width: 900px;
    height: 600px;
  }
  /* 900x450 image */
  .image-900x450 {
    width: 900px;
    height: 450px;
  }
  /* 900x563 image */
  .image-900x563 {
    width: 900px;
    height: 563px;
  }
  /* 450x526.5 image */
  .image-450x526\.5 {
    width: 450px;
    height: 526.5px;
  }
    /* 900x1125 image */
  .image-900x1125 {
    width: 900px;
    height: 1125px;
  }
    /* 900x540 image */
  .image-900x540 {
    width: 900px;
    height: 540px;
  }
  /* Responsive behavior for mobile */
  @media only screen and (max-width: 768px) {
    .image-900x544, .image-900x600, .image-900x450, .image-900x563, .image-450x526\.5 .image-900x1125 {
      width: 100%;
      height: auto;
    }
  }
</style>