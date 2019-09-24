#!/bin/bash

#block(name=train_mff, threads=8, memory=40000, gpus=2, hours=48)
mmskl configs/recognition/mff/kinetics-skeleton/train1.yaml
