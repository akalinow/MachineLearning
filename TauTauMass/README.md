Fetch the data.
```
mkdir data
wget http://akalinow.web.cern.ch/akalinow/MachineLearning/TauTauMass/data/htt_features_train.pkl
wget http://akalinow.web.cern.ch/akalinow/MachineLearning/TauTauMass/data/htt_features_ggH125.pkl
```

Setup the environment
```
mkdir tensorflow
virtualenv --system-site-packages -p python3 tensorflow
source tensorflow/bin/activate
pip install --upgrade tensorflow pandas scikit-learn matplotlib numpy
```

Train the NN (requires TensorFlow, pandas, scikit-learn, matplotlib, numpy)

```
python train.py --max_epoch 500 --model_dir model/1/ --train_data_file data/htt_features_train.pkl --batchSize 64
```

Test on full simulation ggH125 events

```
python plot.py --test_data_file data/htt_features_DY_ggH125.pkl --model_dir model/1/
```

docker run -u $(id -u):$(id -g) -v ${PWD}:/tf/notebooks  -it -p 8888:8888 tensorflow/tensorflow:latest-py3-jupyter

docker run -it  -v ${PWD}:/MachineLearning -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -u $(id -u):$(id -g) tensorflow/tensorflow:latest-py3-jupyter-deps-installed