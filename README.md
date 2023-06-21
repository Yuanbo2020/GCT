# GCT: Gated Contextual Transformer for Sequential Audio Tagging
 
# Noiseme and DCASE2018 datasets

Due to the privacy issues of the Noiseme dataset, we can only release the features and the manual annotations of the Noiseme dataset. The features and the corresponding manually annotated weak and sequential labels can be found <a href="https://github.com/Yuanbo2020/GCT/tree/main/Full_dataset_of_Noiseme" 
target="https://github.com/Yuanbo2020/GCT/tree/main/Full_dataset_of_Noiseme">here</a>.

For the separate files of tagged sequential labels of DCASE and Noiseme datasets, please visit the <a href="https://github.com/Yuanbo2020/GCT/tree/main/Sequential_label_dataset" 
target="https://github.com/Yuanbo2020/GCT/tree/main/Sequential_label_dataset">Sequential_label_dataset</a>.

Please feel free to use the Noiseme dataset, the sequential labels of DCASE and Noiseme datasets and the source code below, and consider citing our paper as

```bibtex
@INPROCEEDINGS{10096842,
  author={Hou, Yuanbo and Wang, Yun and Wang, Wenwu and Botteldooren, Dick},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Gct: Gated Contextual Transformer for Sequential Audio Tagging}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096842}}
```


# Training, inference and evaluation

## 1) GCT: Gated Contextual Transformer

### GCT on the DCASE2018 dataset

```python
Unzip the sys_b64_e500.7z under the GCT_DCASE2018/application folder.
```

##### 1.1 Inference and Evaluation on the DCASE2018 dataset
```python
python evaluate_GCT.py
-------------------------------------------------------------
BLEU:  0.6912116148965892
F-score:  0.927536231884058  AUC: 0.9451526007572187
```
##### 1.2 Training
```python
If you want to train the GCT on the DCASE2018 dataset yourself, 
1) Unzip the Dataset.7z.001 ~ Dataset.7z.031 under the application folder
2) python train_GCT.py
```

### GCT on the Noiseme dataset

```python
Unzip the sys_b64_e500.7z under the GCT_Noiseme/application folder.
```

##### 1.1 Inference and Evaluation on the DCASE2018 dataset
```python
python evaluate_GCT.py
-------------------------------------------------------------
BLEU:  0.3524526737686447
F-score:  0.5287138111058377  AUC: 0.6622533958429383
```
##### 1.2 Training
```python
If you want to train the GCT on the Noiseme dataset yourself, 
1) Unzip the Dataset.7z.001 ~ Dataset.7z.046 under the application folder
2) python Train_GCT.py
```


## 2) CBGRU-GLU-CTC (named GLU-CTC)

##### 1.1 Data preparation

```python
Unzip the sys_64_e500.7z under the GLU-CTC/application folder.
```
##### 1.2 Inference 
```python
Under the path of GLU-CTC/application: python inference_GRU_CTC.py
```
##### 1.3 Evaluation
```python
python evaluate_GRU_CTC.py
-------------------------------------------------------------
F-score:  0.4894166236448116  AUC: 0.5711581205061816  BLEU:  0.28028168048735863
```
##### 1.4 Training
```python
If you want to train the GLU-CTC yourself, 
1) Unzip the Dataset.zip.001 ~ Dataset.zip.016 under the Full_dataset_of_Noiseme folder
2) Copy the unzipped Dataset folder to the GLU-CTC/application
3) python training_GRU_CTC.py
```
##### 1.5 Citation of CBGRU-GLU-CTC (GLU-CTC)

```bibtex
@INPROCEEDINGS{8683627,
  author={Hou, Yuanbo and Kong, Qiuqiang and Li, Shengchen and Plumbley, Mark D.},
  booktitle={ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Sound Event Detection with Sequentially Labelled Data Based on Connectionist Temporal Classification and Unsupervised Clustering}, 
  year={2019},
  volume={},
  number={},
  pages={46-50},
  doi={10.1109/ICASSP.2019.8683627}}
```

## 3) Transformer
```python
Unzip the sys_b64_e500.7z under the Transformer/application folder.
```

##### 1.1 Inference and Evaluation
```python
python evaluate_transformer.py
-------------------------------------------------------------
BLEU:  0.3311125194951529
F-score:  0.46405228758169936  AUC: 0.5896154049693271
```

##### 1.2 Training
```python
If you want to train the Transformer yourself, 
1) Unzip the Dataset.7z.001 ~ Dataset.7z.046 under the application folder
2) python Train_transformer.py
```

## 4) cTransformer (Contextual Transformer)
```python
copy Dataset.7z.001 ~ Dataset.7z.046 from Transformer/application to cTransformer/application 
```

##### 1.1 Training
```python
If you want to train the cTransformer yourself, 
1) Unzip the Dataset.7z.001 ~ Dataset.7z.046 under the application folder
2) python Contextual_Transformer.py
```

##### 1.2 Citation of cTransformer (Contextual Transformer)

```bibtex
@inproceedings{hou22_interspeech,
  author={Yuanbo Hou and Zhaoyi Liu and Bo Kang and Yun Wang and Dick Botteldooren},
  title={{CT-SAT: Contextual Transformer for Sequential Audio Tagging}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4147--4151},
  doi={10.21437/Interspeech.2022-196}
}
```
 
