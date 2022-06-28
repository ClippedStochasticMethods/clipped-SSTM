# clipped-SSTM

This is the code for the experiments from the paper "Near-Optimal High Probability Complexity Bounds for Non-Smooth Stochastic Convex Optimization with Heavy-Tailed Noise". Files:

— graph_tools_lot.py — functions for making the plots

— graphs.ipynb — Jupyter Notebook for making the plots

— synthetic.ipynb — Jupyter Notebook for running the experiments on synthetic data

— heavy_tail_bert_cola.ipynb — Jupyter Notebook for running the experiments on BERT + CoLA task

— heavy_tail_bert_mrpc.ipynb — Jupyter Notebook for running the experiments on BERT + MRPC task

— heavy_tail_bert_sst-2.ipynb — Jupyter Notebook for running the experiments on BERT + SST-2 task

— heavy_tail_bert_stsb.ipynb — Jupyter Notebook for running the experiments on BERT + STS-B task

— heavy_tail_resnet_imagenet.ipynb —  Jupyter Notebook for running the experiments on ResNet + Imagenet-100 task. To run these experiments one needs to download Imagenet-100 dataset

— optimizers.py — implementation of clipped-SSTM and clipped-SGD in PyTorch

— utils.py — helper functions


## Licence
If you want to use our code, please, cite our work:
> @article{gorbunov2021near,
>  title={Near-optimal high probability complexity bounds for non-smooth stochastic optimization with heavy-tailed noise},
>  author={Gorbunov, Eduard and Danilova, Marina and Shibaev, Innokentiy and Dvurechensky, Pavel and Gasnikov, Alexander},
>  journal={arXiv preprint arXiv:2106.05958},
>  year={2021}
> }

