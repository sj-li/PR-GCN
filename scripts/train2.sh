#!/bin/bash

#block(name=train_mff2, threads=2, memory=12000, gpus=1, hours=72)
mmskl configs/recognition/mff/kinetics-skeleton/train2.yaml
