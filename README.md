# RoVLaP: Bridging Learned Visual Perception and Symbolic Belief-Space Planning

This codebase contains the implementation of the RoVLaP benchmark. It is based on the ViPlan benchmark. [ViPlan](https://github.com/merlerm/ViPlan) is a benchmark for planning in Blocksworld and iGibson.


## Project structure

The project is divided into the following main sections:

- Source code: [viplan](viplan/README.md)
- Data: [data](data/README.md)

## Installation

The ViPlan benchmark is made up of several components, including the main experiment code and specific code for the two environments (Blocksworld and Household).

### Experiments

To run the experiments, you need to install the required packages. We recommend using mamba and provide an environment file for easy installation. The virtual environment requirements can be found at `environment.yml`, and it can be created as preferred. Here we report examples using `mamba`.

[Installing mamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

Using `mamba`:

```bash
mamba env create -p ./viplan_env -f environment.yml
mamba activate ./viplan_env
```

> [!WARNING]
> Using conda is not ufficially supported, but if you want, swap mamba with conda everywhere (also in the sh_scripts) and you should be good.
>  e.g.

```bash
conda env create -p ./viplan_env -f environment.yml
conda activate ./viplan_env
```

If you wish to use Flash Attention, it needs to be installed separately with the following command:

```bash
pip install flash-attn --no-build-isolation
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
```

> [!WARNING]
> At the time of writing, Molmo has [an issue](https://huggingface.co/allenai/Molmo-7B-D-0924/discussions/44) with the latest version of `transformers` (>= 4.51.0). To run Molmo, please downgrade `transformers` to version 4.50.3 with `pip install transformers==4.50.3`.

### CPP Policy
The following is required if you're using the CPP Policy:
```bash
mamba install mono -c conda-forge
pip install "git+https://github.com/guyazran/up-cpor@neusCR"
```


### Environments

#### Blocksworld

> **Note:** In RoVLaP we did not use Blocksworld. Skip to iGibson below.

The Blocksworld environment is based on the [Photorealistic Blocksworld](https://github.com/IBM/photorealistic-blocksworld) renderer, which is based on Blender. To install the Blender-based renderer, from the root directory of the repository, run the following commands:

```bash
./setup_blocksworld.sh
```

Additionally, the libxi package needs to be installed (e.g., `sudo apt-get install libxi`) or available in the cluster.

#### iGibson

Here is the list of specific requirements to use iGibson:

- `apptainer` (former Singularity)
- Encription key to be requested at [this link](https://docs.google.com/forms/d/e/1FAIpQLScPwhlUcHu_mwBqq5kQzT2VRIRwg_rJvF0IWYBk_LxEZiJIFg/viewform)

The Household environment is instead based on a custom version of [iGibson](https://github.com/StanfordVL/iGibson). 
To install the environment, first clone our fork of iGibson:

```bash
git clone --depth 1 --single-branch --branch release_viplan https://github.com/f1ren/iGibson.git ./iGibson --recursive
git clone https://github.com/StanfordVL/behavior.git
```

Since iGibson requires specific packages, we recommend running it inside a container. Our code is designed to work with [Apptainer](https://apptainer.org). To pull the image, run:

```bash
apptainer cache clean
apptainer pull docker://igibson/igibson:latest
```
This will create a file called `igibson_latest.sif` (it should take approximately 15 minutes), which is expected to be in the root directory. This file is a Singularity image that contains all the dependencies needed to run iGibson. To open a shell inside the container run:
```bash
apptainer exec --nv igibson_latest.sif bash
```

In the container, install Python3.8:
```bash
mamba install python=3.8
```

Then, install the iGibson dependencies, still inside the container:

```bash
python3.8 -m venv --system-site-packages ./igibson_env
source igibson_env/bin/activate
pip install -e ./iGibson
pip install -e ./behavior
pip install notebook pyquaternion shapely uvicorn fastapi unified_planning
pip install unified_planning[engines]
```

Afterwards, the iGibson custom assets need to be downloaded following the instructions at [this page](https://stanfordvl.github.io/iGibson/dataset.html):

To download the assets, run:

```bash
cd iGibson
wget --no-check-certificate https://storage.googleapis.com/gibson_scenes/ig_dataset.tar.gz
mkdir igibson/data
tar -xzvf ig_dataset.tar.gz -C ./igibson/data
```

Then, still in the iGibson folder, from inside the container run:
```bash
python -m igibson.utils.assets_utils --download_assets
python -m igibson.utils.assets_utils --download_demo_data
```

As some of the assets are encrypted, you will need to download the key provided by the iGibson team. The key can be requested by filling out the form at [this link](https://docs.google.com/forms/d/e/1FAIpQLScPwhlUcHu_mwBqq5kQzT2VRIRwg_rJvF0IWYBk_LxEZiJIFg/viewform) and then needs to be placed inside the `iGibson` folder under `igibson/data/igibson.key`.

After this, the iGibson environment is ready to be used. For the benchmark, we use a client-server architecture, where the server runs inside the container and the client runs in the main execution environment. Scripts are provided in the `sh_scripts` folder to run the server and the client.

## Running Experiments

See [running experiments](./viplan/experiments/README.md) for examples of how we ran the experiments.

## Results

* We include all the results from the experiments reported in the paper in the `results` folder.
* Here's an [example log file](./results/planning/igibson/simple/cpp/gpt-4.1/2026-05-18_13-40-43/execution.jsonl).
* To process the results into a table, use the scripts in `analyze` folder. This reproduces the table reported in the paper.
