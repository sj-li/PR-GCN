#!/bin/bash

#block(name=train_st-gcn, threads=2 memory=11000, gpus=2, hours=48)
mmskl configs/recognition/st_gcn/kinetics-skeleton/train.yaml
