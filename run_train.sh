#!/bin/bash

#block(name=train_st_gcn, threads=1, memory=40000, gpus=4, hours=24)
mmskl configs/recognition/st_gcn/kinetics-skeleton/train.yaml
