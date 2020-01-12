#!/bin/bash

#block(name=cst-gcn, threads=8, memory=45000, gpus=4, hours=480)
mmskl configs/recognition/cst_gcn/kinetics-skeleton/train.yaml
