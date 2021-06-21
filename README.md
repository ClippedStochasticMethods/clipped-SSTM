# clipped-SSTM

This is the code for the experiments from the paper "Near-Optimal High Probability Complexity Bounds for Non-Smooth Stochastic Optimization with Heavy-Tailed Noise" by Eduard Gorbunov, Marina Danilova, Innokentiy Shibaev, Pavel Dvurechensky, Alexander Gasnikov: https://arxiv.org/pdf/2106.05958.pdf. 

Files:

— graph_tools_lot.py — functions for making the plots

— graphs.ipynb — Jupyter Notebook for making the plots

— heavy_tail_bert_cola.ipynb — Jupyter Notebook for running the experiments on BERT + CoLA task

— heavy_tail_resnet_imagenet.ipynb —  Jupyter Notebook for running the experiments on ResNet + Imagenet-100 task. To run these experiments one needs to download Imagenet-100 dataset

— optimizers.py — implementation of clipped-SSTM and clipped-SGD in PyTorch

— utils.py — helper functions
