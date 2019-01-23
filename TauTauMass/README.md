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
python train.py --max_epoch 500 --model_dir model/1/ --train_data_file data/htt_features_train.pkl --batchSize 4096
```

Test on full simulation ggH125 events

```
python plot.py --test_data_file data/htt_features_ggH125.pkl --model_dir model/1/
```

