# Ecological Sampling Python

The code is a simple Demo of the Ecological Sampling (ES) method,
which generates gaze shifts on video clips (frame sequences).
It is a baseline implementation of the Ecological Sampling model described in
```Boccignone & Ferraro [1]```, a stochastic model of eye guidance
The gaze shift mechanism is conceived as  an active random sampling  that
the "foraging eye" carries out upon the visual landscape,
under the constraints set by the  observable features   and the
global complexity of the  landscape.
The actual  gaze relocation is  driven by a stochastic differential equation
whose noise source is sampled from a mixture of <img src="https://render.githubusercontent.com/render/math?math=\alpha">-stable distributions.
The sampling strategy  allows to mimic a fundamental property of  eye guidance:
where we choose to look next at any given moment in time is not completely deterministic,
but neither is it completely random

The experiment consists in putting into action an  artificial observer, generating a visual scanpath
(a sequence of fixations, saccades or smooth pursuit) on an image sequence using a simple implementation of the
ES method described in ```Boccignone & Ferraro [1] & [2]```.


### Installation

To  create the software library and run the demos:

1) unpack the compressed zip file in your working directory and cd to such directory (ecosampling)

you will find the following directories:

> - /doc: 		the reference papers
> - /intpoints            function for sampling interest points
> - /protobj              functions for sampling proto-object parameters
> - /Datasets: 		image sequences  to be processed
> - /results: 		to store  results
> - /saltools:            the tools for computing saliency: for demo purposes here you will find the 3-rd party Self-Resemblance method by ```Seo and Milanfar, Journal of Vision (2009) 9(12):15, 1D27```. Store in this directory other salience methods you develop or download from external sources
> - /stats:               statistics tools borrowed from various parties
> - /visualization: 	some visualization tools

2) add the path to this directory and subdirectories in your Matlab environment

3) edit if you like the /config/config_simple.m script for tuning the parameters of the experiment
or just try it in the proposed configuration. Such configuration script will be useful to you  because
it holds  all settings used in all parts of the code, enabling the exact
reproduction of the experiment at some future date

4) run demo program
```
runEcologicalSampling
```

### Image sequences

A sample sequence for demo purpose is provided with the source code in the Datasets directory
Put in this directory your own video clips, as sequences of frames

### Demo program

The script
```
runEcologicalSampling
```

1) sets the configuration script filename for setting the experiment
2) sets the number of observers you want to simulate
3) generates a scanpath on a video clip for each observer by calling the main function ```esGenerateScanpath()```

```
- esGenerateScanpath():
```
   This is the main function to generate and show the simulation of the model on a video clip
   It generates a visual scanpath, that is a sequence of gaze shifts (saccades and smooth pursuit) on a video sequence
   It is a baseline implementation of the Ecological Sampling model steps, as described in ```Boccignone & Ferraro [1]```
   If the ```VISUALIZE_RESULTS``` variable is set to true in the config script, the maps obtained at the different steps
   of the method are shown at each video frame.

   See the comments in each routine for details of what it does
   Settings for the experiment should be held in the configuration
   file.

```
- esComputeFeatures()
```
  The function is a simple wrapper for feature computation. Executes some kind
  of feature extraction algorithm which is defined from the parameter
  fType by calling the appropriate function.
  Here for simplicity only the Self Resemblance features extraction method has been considered.
  Actual feature computation is performed by ```ThreeDLARK()``` method, see the directory ```./saltools/SelfResemblance2/```
  The SR method provides comparable performance to other methods but at
  a lower computational complexity and most important can deal with a moving camera
  If other methods need to be experimented, then you should extend the ```if...elseif...end```
  control structure

```
- esComputeSaliency():
```
   The function is a simple wrapper for salience computation. Executes some kind
   of salience computation algorithm which is defined from the parameter
   ```salType``` by calling the appropriate function. Here for simplicity only
   the 3-D SELF RESEMBLANCE SPATIO TEMPORAL SALIENCY method has been considered.
   Actual salience computation is performed by ```SpaceTimeSaliencyMap()``` method, see the directory ```./saltools/SelfResemblance2/```
   The SR method provides comparable performance to other methods but at
   a lower computational complexity and most important can deal with a moving camera
   If other methods need to be experimented, then you should extend the ```if...elseif...end```
   control structure


```
- esSampleProtoMap():
```
   Generates the patch map or proto-object map <img src="https://render.githubusercontent.com/render/math?math=M(t)">

```
- esSampleProtoParameters():
```
   Generates the patch map <img src="https://render.githubusercontent.com/render/math?math=M(t)"> parameters <img src="https://render.githubusercontent.com/render/math?math=\theta_p">, in terms of maximum likelihood estimation
   of an elliptical approximation of each patch.
   See ```fitellip()``` in ```/protobj```

```
- InterestPoint_Sampling()
```
   Samples ```Interest_Point.Max_Points``` points from set of points (salience map or proto-objects),
   weighted according to their salience

```
- esGetGazeAttractors()
```
   Function computing possible ONE or MULTIPLE gaze attractors
   If a landscape of proto-objects is given then their centers are used as described in ```[1]```
   Otherwise attractors are determined through the IPs sampled from saliency as in ```[2]```.

```
- esComputeComplexity()
```
   Computes spatial configuration complexity <img src="https://render.githubusercontent.com/render/math?math=C(t)"> of Interest points
   The function is a simple wrapper for complexity computation. Executes some kind
   of complexity algorithm which is defined from the parameter
   cType by calling the appropriate function.
   Default is the Shiner-Davison-Landsberg (SDL) complexity ```(Physical review E, 59(2), 1459-1464, 1999)```

```
- esHyperParamUpdate()
```
   Computes the new Dirichlet hyper-parameter <img src="https://render.githubusercontent.com/render/math?math=\nu_{k}(t)">
   Given the complexity <img src="https://render.githubusercontent.com/render/math?math=\mathcal{C}(t)">,  we partition the complexity range in order to define
   <img src="https://render.githubusercontent.com/render/math?math=K"> possible complexity events <img src="https://render.githubusercontent.com/render/math?math=\{E_{\mathcal{C}(t)}=k\}_{k=1}^{K}">.
  This way the hyper-parameter update    can be rewritten as the recursion
        <img src="https://render.githubusercontent.com/render/math?math=\nu_{k}(t)= \nu_k(t-1) +\left[ E_{\mathcal{C}(t)} = k \right], k=1,\cdots,K">.

```
- esGazeSampling()
```
   Function computing, using gaze attractors,  the actual  gaze relocation by  sampling the appropriate
   noise parameters from a mixture of <img src="https://render.githubusercontent.com/render/math?math=\alpha">-stable distributions as a function of the oculomotor state
   The parameters are used to propose one or multiple candidate gaze shifts actually implemented by
   calling for ```esLangevinSimSampling()```, which performs the Langevin step.
   Eventually decides and sets the actual shift to the new Focus of Attention

```
- esLangevinSimSampling()
```
   performing one step of the Langevin like stochastic differential equation
   for determining the gaze shift

```
- stabrnd():
```
   This is the actual procedure for sampling from <img src="https://render.githubusercontent.com/render/math?math=\alpha">-stable distributions and implements the CMS method:
   ```Stable Random Number Generator. Based on the method of J.M. Chambers, C.L. Mallows and B.W. Stuck, "A Method for Simulating Stable Random Variables," JASA 71 (1976): 340-4.```
   Located in ```./stats/alphastable/CMS``` directory

### Tips

Different gaze shifting behaviors can be obtained by playing with parameters in the configuration script

## Citing

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



