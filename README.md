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
   and `INSTALL.md`) into the `pylocaltie` environment. 
   In the gnsstk directory: 
   ```
   ./build.sh -e -i $CONDA_PREFIX -j $(nproc) -- -DCMAKE_BUILD_TYPE=release
   ```
   NB: This may be fixed in gnsstk, but the last time I tried this I had to do
   ```
   sed -i 's/-DGNSSTK_EXPORT /-DGNSSTK_EXPORT= /' swig/CMakeLists.txt
   CXXFLAGS="-include cstdint -Wno-deprecated" \
   ./build.sh -c -e -i $CONDA_PREFIX -j $(nproc) -- -DCMAKE_BUILD_TYPE=release
   ```
   Due to bugs in the gnsstk build system.
   You also may need to install swig, cmake, etc. into your conda environment if they aren't already on your machine:
   ```
   conda install -c conda-forge cmake swig<4.3 compilers make
   ```
   (SWIG 4.5 broke the RINEX header generation in vdif2rinex. I can confirm
   that 4.0.3 works)
