## ML model for track reconstruction in WAWTPC detector

This package is used for training a NN for track reconstruction in the
[WAWTPC](https://indico.cern.ch/event/1104299/contributions/5058543/attachments/2534412/4361430/cwiok_slides_EuNPC2022_v3.pdf) 
detector.

The package uses TensorFlow and other packages. All the necessary packages are available in a 
[akalinow/tensorflow-gpu](https://hub.docker.com/r/akalinow/tensorflow-gpu) container. It is best to use singularity to run it:

```Bash
apptainer run --nv --bind work_dir_on_host:work_dir_in_container docker://akalinow/tensorflow-gpu
```
and use a Jupyter available inside the container:

```Bash
jupyter lab --no-browser --ip=0.0.0.0 --notebook-dir=$HOME
```

### The coordinates:

The cartesian coordinates of the detector are as follows:

* **X** - along the plane in the half of the detector heights.
* **Z** - vertical axis, pointing downwards. The Z coordinate values are arbitrary 
          (due to details how it is measured), and do not correspond to real height within the detector
* **Y** - third axis closing a right handed cartesian coordinate system          


## Input data:

* the input data contains events with $\alpha$ and $^{12}_{6}C$ ions tracks from a detector simulation.
  The particles originate from a common vertex, and fly almost (but not exactly) in opposite directions.
  The vertex position is restricted to a narrow cylinder around the X axis, and span the whole detector length.

* three 2D images, resolution: **521x256**, pixel value: **arbitrary integer**, can be negative.
  Each image is called a U,V,W projection according to the name of the second projection axis. 
  First axis is common to all projections, and is called Z axis. The actual projections heights differ between U, V, W, and are zero padded to 256 rows.

* the width and height dimensions are scaled in discrete detector coordinates: 
    * readout strip number for U,V,W
    * time bin number for T
  the transformation from (U,V,W,T) to cartesian (X,Y,Z) coordinates depends on the  
  detector working point. Currently the parameters of this transformation are hard coded in
  [XYZtoUVWT(data)](utility_functions.py#L46-L58)  function.
  
 ## Target:
 
* 3D cartesian positions of the vertex, and tracks endpoints, **9** numbers in total:

  $X_{\text{vertex}}, Y_{\text{vertex}}, Z_{\text{vertex}}$  
  $X_{\alpha~\text{end point}}, Y_{\alpha~\text{end point}}, Z_{\alpha~\text{end point}}$, <br>
  $X_{\text{carbon~end point}}, Y_{\text{carbon~end point}}, Z_{\text{carbon~end point}}$

## Code organisation:

The code is organised in a set of function in python modules:

* [io_functions.py](io_functions.py) - functions for reading and preprocessing the data. The data is originally stored in [ROOT TTree](https://root.cern.ch/doc/master/classTTree.html) file format. This format is commonly used in particle energy physics community. The ROOT TTree is loaded into numpy with use of [uproot](https://uproot.readthedocs.io/en/latest/index.html) library in forma of a generator function: [generator(files, batchSize, features_only=False)](io_functions.py#L106-L131).
The data from generator is injected into 
[tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) object:
```Python
datasetGenerator = partial(io.generator, files=train_files, batchSize=batchSize)

train_dataset = tf.data.Dataset.from_generator(
     datasetGenerator,
     output_signature=(
         tf.TensorSpec(shape=(batchSize,) + (io.projections.shape), dtype=tf.float32),
         tf.TensorSpec(shape=(batchSize,9), dtype=tf.float32)))
```
**Note:** This setup is suboptimal. 90\% of the learning time is spent in data reading. Most likely translation into [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format will bring improvements.

* [utility_functions.py](utility_functions.py) - set of various helper functions. Mostly used for filling a Pandas dataset with model result for performance studies.

* [plotting_functions.py](plotiing_functions.py) - set of functions used to make plots for the input data.
  For example the [plotEvent(data, model)](plotting_functions.py#L42C5-L77) function plots an input images with target, and model result endpoints (after transforming from (X,Y,Z) to (U,V,W,T) representation).
  
The actual work is done with use of Jupyter notebooks:
* [WAWTPC_ML.ipynb](WAWTPC_ML.ipynb) - cells with train, and test data loading, model architecture definition, training, performance plots on the test data

* [WAWTPC_analysis_ML_reco.ipynb](WAWTPC_analysis_ML_reco.ipynb) - cells for loading data from a real experiment. Data From a real experiment does not contain the true positions of the endpoints, so most of the performance plots are meaningless. The most important plots are **input images with model result overlaid.**

* [ROOT_to_TFRecord.ipynb](ROOT_to_TFRecord.ipynb) - Cells with experiments on the I/O. Work in progress.

* [WAWTPC_analysis_classic_reco.ipynb](WAWTPC_analysis_classic_reco.ipynb) - cells for loading and plotting results from a classic algorithm. The data from classic algorithm is stored in a ROOT TTree with different format than trees used for ML algorithm. The classic algorithm works as follows:
    * find lines in each 2D plot, using a Hough transform
    * combine line direction into a 3D line proposal
    * find a 3D line orientation minimising transverse distance from pixels to the line
    * calculate the value density along a 3D line by combining information from all 2D images
    * find line endpoints by fitting an expected density along a line

The classic algorithm works well for about 95\% of cases, with significant failure for the rest.

**We are looking for students to continue and extend this study**

If you are interested please contact [Artur.Kalinowski@fuw.edu.pl](Artur.Kalinowski@fuw.edu.pl).