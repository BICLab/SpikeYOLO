The purpose of this document is to illustrate the training method of SpikeYOLO on Gen1 data(GEN1's experiments were performed based on spikingjelly).
## 1. Data download:
https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/

## 2. preprocessing: python pre_gen1.py
Refer to environmental requirements “https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/”

## 3. install spikingjelly
cd spikingjelly-0.0.0.0.12
python setup.py install

## 4.train
python train.py

## 5.test / get_firing_rate
python test.py