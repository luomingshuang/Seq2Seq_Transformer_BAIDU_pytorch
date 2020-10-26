# Seq2Seq_Transformer_BAIDU_pytorch
Introduction
----
This is a project for seq2seq lip reading on a lip-reading dataset called BAIDU (not published) with 
transformer model.
In this project, we implemented it with Pytorch.

Dependencies
----
* Python: 3.6+
* Pytorch: 1.3+
* Others

Dataset
----
This project is trained on BAIDU (grayscale).

Training And Testing
----
About the modeling units in this work, we built our vocabulary words table as following:
```
{0: '<sos>', 1: '<eos>', 2: 'dao', 3: 'hang', 4: 'she', 5: 'zhi', 6: 'chong', 7: 'xin', 8: 'bo', 
9: 'fang', 10: 'da', 11: 'guan', 12: 'bi', 13: 'zuo', 14: 'chuang', 15: 'kai', 16: 'shou', 
17: 'yin', 18: 'ji', 19: 'wei', 20: 'deng', 21: 'zhu', 22: 'tiao', 23: 'gao', 24: 'wen', 
25: 'du', 26: 'jie', 27: 'ting', 28: 'dian', 29: 'hua', 30: 'yu', 31: 'hao', 32: 'GPS', 
33: 'huan', 34: 'tai', 35: 'zhuan', 36: 'jin', 37: 'guang', 38: 'shang', 39: 'yi', 40: 'sheng', 
41: 'kong', 42: 'di', 43: 'CD', 44: 'xiao', 45: 'liang', 46: 'ma', 47: 'you', 48: 'xiang', 
49: 'qi', 50: 'niao', 51: 'kan', 52: 'tu', 53: 'zan', 54: 'yue', 55: 'xia', 56: 'mo', 57: 'ni', 
58: 'gua', 59: 'duan', 60: 'dong', 61: 'xi', 62: 'tong', 63: 'ge', 64: 'tian', 65: 'que', 
66: 'ren', 67: 'jiang', 68: 'yuan', 69: 'shi', 70: 'mu', 71: 'liu', 72: 'lan', 73: 'hou', 
74: 'xun', 75: 'zhao', 76: 'xing', 77: 'qu', 78: 'shu', 79: 'che', 80: 'jiu', 81: 'jia', 82: 'feng', 83: 'xuan', 84: 'suo', 
85: 'zen', 86: 'me', 87: 'yang', 88: 'li', 89: 'lu', 90: 'jian', 91: 'xie', 92: 'wo', 93: 'cha'}
``` 
First, we need to config the parameters in the python file called "config.py". We also need to change 
the "pre_process.py" according to the entails about our data. 
And we preprocess the data with the following commands:
```
python pre_process.py  ##We can get BAIDU.pickle
python collect_char_list.py  ##We can get char_list.pkl
python ngram_lm.py  ##We can get bigram_freq.pkl
```
Then, we can set the number of GPUs for training according to our realistic devices. 
If we want to use 4 GPUs to train, we can change the parameters in "train.py", such as "model = nn.DataParallel(model, device_ids=[0,1,2,4])".
We can train our model with the following command:
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python train.py 
```
Last, when our training loss is converged, we can get the model called "BEST_checkpoint_6_words.tar".
Here, we provide a final model which is available at [GoogleDrive](https://drive.google.com/drive/folders/1rP2t2BcalpEwXFKxd9gVYc4qXgh5nxQ_).
And copy the checkpoint to this folder. Here, we provide the beam search method with ngram_language model.
We can test the model as follows:
```
##When we test the model without language model, we set the beam size in "test_LM.py" to 1.
python test_LM.py
##When we test the model with language model, we set the beam size in "test_LM.py" to 2 (3,4,5).
python test_LM.py
```
Our testing results as follows:
```
beam size=1, WER=10.63% (Baseline)
beam size=2, WER=9.62%
beam size=3, WER=9.22%
beam size=4, WER=9.04%
beam size=5, WER=8.97%
```