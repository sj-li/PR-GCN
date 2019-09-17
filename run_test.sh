#!/bin/bash

#block(name=test_st_gcn, threads=1, memory=1000, gpus=4, hours=24)
#mmskl configs/recognition/st_gcn/kinetics-skeleton/test.yaml --checkpoint work_dir/recognition/st_gcn/kinetics-skeleton/latest.pth
mmskl configs/recognition/st_gcn/kinetics-skeleton/test.yaml --checkpoint checkpoints/st_gcn.kinetics-6fa43f73.pth
