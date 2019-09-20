#!/bin/bash

#block(name=train_mff, threads=8, memory=40000, gpus=4, hours=24)
mmskl configs/recognition/mff/kinetics-skeleton/train.yaml
