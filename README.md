# Ecological Sampling Python

<img src="https://raw.githubusercontent.com/phuselab/INMCA_Nobre/main/docs/source/_static/logo.svg?token=GHSAT0AAAAAABUT73UR3IGPVZ6SA4HE2SFGYVBYNTQ">

The code is a simple demo of the Ecological Sampling (ES) method, which generates gaze shifts on video clips (frame sequences). It is a baseline implementation of the Ecological Sampling model described in
```Boccignone & Ferraro [1]```, a stochastic model of eye guidance
The gaze shift mechanism is conceived as  an active random sampling that
the "foraging eye" carries out upon the visual landscape, under the constraints set by the observable features and the global complexity of the landscape.
The actual gaze relocation is driven by a stochastic differential equation
whose noise source is sampled from a mixture of alpha-stable distributions.

The sampling strategy allows to mimic a fundamental property of eye guidance:
where we choose to look next at any given moment in time is not completely deterministic,
but neither is it completely random

The experiment consists in putting into action an artificial observer, generating a visual scanpath
(a sequence of fixations, saccades or smooth pursuit) on an image sequence using a simple implementation of the
ES method described in ```Boccignone & Ferraro [1] & [2]```.

## Requirements

The following are a list of basic requirements and tested OS for the ecosampling execution.

- **Python:** Tested on >= 3.8, but might work on previous versions as well
- **Tkinter:** A GUI backend for matplotlib
- **macOS Monterey:** Tested on version 12.4
- **Ubuntu 20:** Tested on version

### Installing Tkinter

- **Ubuntu via apt-get**
```bash
sudo apt-get install python3-tk
```

- **macOS via Brew**
```bash
sudo apt-get install python3-tk
```

## Installation

To create the software library and run the demos, given that your python is in the correct version and Tkinter is installed, clone the repository, enter the project root folder and install the project requirements:

```bash
$ git clone https://github.com/phuselab/INMCA_Nobre
$ cd INMCA_Nobre/
$ pip install -r requirements.txt
```

## Adding your data

A sample sequence for demo purpose is provided with the source code in the ``data`` directory. To add your own clips add a folder in the same directory.

The folder that you add need to have the video as a sequence of ordered frames. To direct the execution of the application for your datafolder, change the configurations in ``ecosampling/config.py``.

## Usage

Ecosampling might be used as a application stand alone or as an API to write your own gaze sampling script. Following are the methods for execution the application

### Running as an Application

To run the application as is you can run the following command on the root of the project:

```bash
python ecosampling
```
This should be enough to generate all the gaze sampling and the fields of attention for your application.

If you want to plasy around, change parameters, or try with your personalized data set, try changing the ``ecosampling/config.py`` file. Different gaze shifting behaviors can be obtained by playing with parameters in the configuration script.

## Documentation - Read More

If you want in-depth understanding of how the project is organized or on the functioning of the API read the documentation provided as an Read the Docs website.

To open the web documentation, open the file on the directly: ``docs/build/index.html``.


## Citing

To cite the current work follow the BibTeX formats:

```[1] G. Boccignone and M. Ferraro, Ecological Sampling of Gaze Shifts```

```
@article{BocFerSMCB2013,
   title={Ecological Sampling of Gaze Shifts},
   author =  "Boccignone, Giuseppe and Ferraro, Mario",
   journal="{IEEE} Trans. Systems Man Cybernetics - B",
   year={2013},
   pages={1-1},
   url = "http://dx.doi.org/10.1109/TCYB.2013.2253460",
}
```

```[2] G. Boccignone and M. Ferraro, The active sampling of gaze-shifts```

```
@incollection{BocFerIciap2011,
   author = "Boccignone, G. and Ferraro, M.",
   title = {The Active Sampling of Gaze-Shifts},
   booktitle = {Image Analysis and Processingâ ICIAP 2011},
   series = {Lecture Notes in Computer Science},
   editor = {Maino, Giuseppe and Foresti, Gian},
   publisher = {Springer Berlin / Heidelberg},
   pages = {187-196},
   volume = {6978},
   year = {2011}
}
```



