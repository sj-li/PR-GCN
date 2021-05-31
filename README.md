# PR-GCN

## Introduction

This repository provides the official PyTorch implementation for the following paper:

**Pose  Refinement  Graph  Convolutional  Network  for  Skeleton-basedAction  Recognition**<br>
[Shijie Li](https://sj-li.com/),  Jinhui Yi, Yazan Abu Farha, [Juergen Gann](http://gall.cv-uni-bonn.de/)<br>
In RA-L with ICRA 2021 .<br>
[**Paper**](https://arxiv.org/abs/2007.12072)
> **Abstract:** *With the advances in capturing 2D or 3D skeleton data, skeleton-based action recognition has received an increasing interest over the last years. As skeleton data is commonly represented by graphs, graph convolutional networks have been proposed for this task. While current graph convolutional networks accurately recognize actions, they are too expensive for robotics applications where limited computational resources are available. In this paper, we therefore propose a highly efficient graph convolutional network that addresses the limitations of previous works. This is achieved by a parallel structure that gradually fuses motion and spatial information and by reducing the temporal resolution as early as possible. Furthermore, we explicitly address the issue that human poses can contain errors. To this end, the network first refines the poses before they are further processed to recognize the action. We therefore call the network Pose Refinement Graph Convolutional Network. Compared to other graph convolutional networks, our network requires 86\%-93\% less parameters and reduces the floating point operations by 89%-96% while achieving a comparable accuracy. It therefore provides a much better trade-off between accuracy, memory footprint and processing time, which makes it suitable for robotics applications. *

**Instalation:**

``` shell
python setup.py develop
```

**Basic usage:**

Any application in mmskeleton is described by a configuration file. That can be started by a uniform command:
``` shell
python run.py $CONFIG_FILE [--options $OPTHION]
```
which is equivalent to
```
mmskl $CONFIG_FILE [--options $OPTHION]
```
Optional arguments `options` are defined in configuration files.
You can check them via:
``` shell
mmskl $CONFIG_FILE -h
```

### Data Preparation

We experimented on two skeleton-based action recognition datasts: **Kinetics-skeleton** and **NTU RGB+D**.
Before training and testing, for the convenience of fast data loading,
the datasets should be converted to the proper format.
Please download the pre-processed data from
[GoogleDrive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb)
and extract files with
```
cd pr-gcn
unzip <path to pr-gcn-processed-data.zip>
```

### Training

To train a PR-GCN model, run

``` shell
mmskl configs/recognition/pr_gcn/$DATASET/train.yaml [optional arguments]
```

The usage of optional arguments can be checked via adding `--help` argument.
All outputs (log files and ) will be saved to the default working directory.
That can be changed by modifying the configuration file
or adding a optional argument `--work_dir $WORKING_DIRECTORY` in the command line.

### Evaluation

After that, evaluate your models by:

``` shell
mmskl configs/recognition/pr_gcn/$DATASET/test.yaml --checkpoint $CHECKPOINT_FILE
```

## License
The project is release under the [Apache 2.0 license](./LICENSE).

## Acknowledgments
The code is greatly inspired by [mmskeleton](https://github.com/open-mmlab/mmskeleton)


## Citation
Please cite the following paper if you use this repository in your reseach.
```
@ARTICLE{9345415,
  author={S. {Li} and J. {Yi} and Y. {Abu Farha} and J. {Gall}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Pose Refinement Graph Convolutional Network for Skeleton-based Action Recognition}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LRA.2021.3056361}}
```

## Contact
For any question, feel free to contact
```
Shijie Li     : lishijie@iai.uni-bonn.de
```
