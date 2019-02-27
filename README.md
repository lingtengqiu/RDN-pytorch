# RDN
This project aims at providing a fast, modular reference implementation for super-resolution  models using pytorch  
![train loss](result/newplot.png)  
For training you shall download the DIV2k dataset :https://data.vision.ee.ethz.ch/cvl/DIV2K/ and put your train_img,and valid_img to the DIV2K_train_HR and DIV2K_valid_HR. <br> 
First you shall python main.py process to generate downsample data and then you can train your RDN-Net to use python main.py train .<br>
Second,you can change your para from the config.py All of it realized from pytorch.<br>
Finaly if you want to see the output ,you can download the visdom to see output real time
if you have question ,email me 1259738366@qq.com
