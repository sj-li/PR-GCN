#!/bin/bash

#block(name=train_mff3, threads=2, memory=40000, gpus=2, hours=48)
mmskl configs/recognition/mff/kinetics-skeleton/train3.yaml
