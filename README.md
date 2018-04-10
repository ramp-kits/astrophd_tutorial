# Deep learning tutorial for Astro PhDs - Galaxy deblending

[![Build Status](https://travis-ci.org/ramp-kits/astrophd_tutorial.svg?branch=master)](https://travis-ci.org/ramp-kits/astrophd_tutorial)

_Authors: Alexandre Boucaud, Marc Huertas-Company & Bertrand Rigaud_

#### Set up

1. clone this repository
  ```
  git clone https://github.com/ramp-kits/astrophd_tutorial
  cd astrophd_tutorial
  ```
  
2. install the dependancies
  - with [conda](https://www.anaconda.com/download/)
  ```
  conda install -y -c conda conda-env     # First install conda-env
  conda env create                        # Use environment.yml to create the 'astrophd_tutorial' env
  source activate astrophd_tutorial       # Activates the virtual env
  ```
  - without `conda` (best to use a virtual environment)
  ```
  python -m pip install -r requirements.txt
  ```

3. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```

4. download the data
  ```
  python download_data.py        # quick-test data for testing ~16Mo
  python download_data.py full   # full dataset ~1.5Go
  ```

#### Local notebook

Get started on this RAMP with the [dedicated notebook](astrophd_tutorial_starting_kit.ipynb)

#### Help

- Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](http:www.ramp.studio) ecosystem.
- Sign up on the [CDS Slack](cds-upsay.slack.com) and join the `#phd_tutorial` channel.


#### Acknowledgements

We thank the CCIN2P3 for providing us with GPU time for the students to train their models during the whole week of the teaching course.

 [![CCIN2P3](img/logosimpleCC.jpg)](https://cc.in2p3.fr/en/)
