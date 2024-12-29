# GAT-CADNet

this code is based on the paper published on CVPR2022 named [GAT-CADNet: Graph Attention Network for Panoptic Symbol Spotting in CAD Drawings](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_GAT-CADNet_Graph_Attention_Network_for_Panoptic_Symbol_Spotting_in_CAD_CVPR_2022_paper.pdf)

Because this paper did not provide official code, I reproduced the model and training process based on the content of 
the paper. Welcome every one to ask me questions and point out the problems that exist.

# Getting Started

## Environment Construction

We recommend users to use `conda` to install the running environment. We train the model on the Ubuntu22.04 
operating system. the version of python is python3.11

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This paper uses an open-source dataset called FloorPlanCAD to train the model. The open-source dataset can be downloaded
through this link [FloorPlanCAD DataSet](https://floorplancad.github.io/)

You can also get the dataset through the following command.

```bash
python data/download_data.py
```

## Model Training

Because this paper only use one kind of parameters to train the model. So we temporarily don't offer the version that 
users can modify the parameters through a config file. If you want to do, you can go to the `main.py` file to modify.

You can use the following command to train the model.

```bash
python main.py
```

# Acknowledgements

If you want to know more about the code, you can read the following paper. And If you find somewhere wrong in our code.
Please give us issues to tell us, Thanks!

```
@inproceedings{zheng_gat-cadnet_2022,
	title = {{GAT}-{CADNet}: Graph Attention Network for Panoptic Symbol Spotting in {CAD} Drawings},
	author = {Zheng, Zhaohua and Li, Jianfang and Zhu, Lingjie and Li, Honghua and Petzold, Frank and Tan, Ping},
	pages = {11737--11746},
	year = {2022},
}
```


