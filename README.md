for train you show download the DIV2k dataset :https://data.vision.ee.ethz.ch/cvl/DIV2K/
and put your train_img,and valid_img to the DIV2K_train_HR and DIV2K_valid_HR
and first you shall python main.py process to generate downsample data
and then you can train your RDN-Net to use python main.py train
you can change your para from the config.py
all of it realized from pytorch 
if you want to see the train out ,you must download the visdom to see 
here you can pip install visdom 
and then python -m visdom.server
if you have question ,email me 1259738366@qq.com
