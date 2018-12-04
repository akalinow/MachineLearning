
Train the NN (requires TensorFlow, Pandas, sklearn, matplotlib, numpy)
```
python train.py --max_epoch 500 --model_dir model/1/ --train_data_file data/train/train.csv
```

Test 

```
python plot.py --test_data_file data/test/test.csv --model_dir model/1/
```

Test on the training data

```
python runModel.py --test_data_file data/train/train.csv
diff -y modelResult.csv test_from_train.csv | grep "|" | wc -l
```
