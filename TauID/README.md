
Train the NN (requires TensorFlow, Pandas, sklearn, matplotlib, numpy)
```
python train.py --max_epoch 500 --model_dir model/1/ --train_data_file data/htt_features_test.pkl
```

Test 

```
python plot.py --test_data_file data/htt_features_test.pkl --model_dir model/1/
```

