# pylocaltie

## Setup

1. Create the conda environment and activate it:
   ```
   conda env create -f pylocaltie.yaml
   conda activate pylocaltie
   ```
2. Install this repo in editable mode so shared modules (e.g. `single_diff_tools.py`)
   are importable from every script, in every subdirectory, regardless of your
   working directory:
   ```
   pip install -e .
   ```
3. LAMBDA dependency: ambiguity resolution uses the TU Delft **LAMBDA** toolbox
   (Psychas et al., Python implementation), which is third-party code and is not
   redistributed in this repo. Download it from the official TU Delft page:
   https://www.tudelft.nl/en/ceg/about-faculty/departments/geoscience-remote-sensing/research/lambda/lambda
   and place `LAMBDA.py` in this repo's top-level directory, next to
   `single_diff_tools.py`.
4. gnsstk dependency: several scripts use **gnsstk**
   (https://github.com/SGL-UT/gnsstk), the Applied Research Laboratories / UT
   Austin GNSS toolkit. It has no public PyPI or conda-forge distribution, so
   `pylocaltie.yaml` does not install it. Build and install the Python
   bindings yourself following the instructions in that repo (`PYTHON.md`
   and `INSTALL.md`) into the `pylocaltie` environment. If you're on ARL:UT's
   network, its internal artifactory
   (`[redacted-internal-host]/artifactory/api/conda/conda-toolkits/stable`)
   also carries prebuilt conda packages.
