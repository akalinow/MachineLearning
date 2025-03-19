## TauTauMass ML model with MET smearing.

The training and validation data is a bare Pythia H->tau tau with H mass in range [50,300].
The ROOT ntuples are created with [RootAnalysis/Pythia8Interface](https://github.com/akalinow/RootAnalysis/tree/devel_AK/HTauTau/Pythia8Interface)
package, and processed with [RootAnalysis/SVFit](https://github.com/akalinow/RootAnalysis/tree/devel_AK/HTauTau/SVfit) package.

The ROOT ntuples are transformed then into Pandas DataFrame parquet files with
[Data_preparation](Data_preparation.ipynb) notebook. The notebook requires PyROOT and relevand packages.
This notebook can be run with a [akalinow/root-fedora31](https://hub.docker.com/repository/docker/akalinow/root-fedora31] Docker container.

The ML is made using the [Training_categorisation](Training_categorisation.ipynb) notebook. This notebook can be run with
a [akalinow/tensorflow-gpu](https://hub.docker.com/repository/docker/akalinow/tensorflow-gpu) Docker container.

The test data is a full simulation of the CMS detector response for a H->tau tau with mass 125 GeV. 

The data files are available on Google drive: [link](https://drive.google.com/drive/u/1/folders/1c8CLaBtkunZwFS_2qEUzvQfKOg1rj9ZK)

## Setup the environment:
```
sudo docker run ---gpus all -rm -it -p 8000:8000 --user $(id -u):$(id -g) -v /scratch_on_host:/scratch akalinow/tensorflow-gpu:latest
```

Inside the container:

```
cd
./start-jupyter.sh
```

Then open the jupyter on URL given in the terminal window, and run the [Training_categorisation](Training_categorisation.ipynb) notebook.



