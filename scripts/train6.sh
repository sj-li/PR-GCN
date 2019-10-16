#!/bin/bash

#block(name=train_mff6, threads=2, memory=22000, gpus=2, hours=48)
mmskl configs/recognition/mff/kinetics-skeleton/train6.yaml
