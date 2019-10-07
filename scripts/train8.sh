#!/bin/bash

#block(name=train_mff8, threads=4 memory=11000, gpus=2, hours=48)
mmskl configs/recognition/mff/kinetics-skeleton/train8.yaml
