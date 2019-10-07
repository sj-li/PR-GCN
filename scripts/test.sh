#!/bin/bash

#block(name=test_st_gcn, threads=1, memory=1000, gpus=1, hours=24)
mmskl configs/recognition/mff/kinetics-skeleton/test.yaml --checkpoint /home/lishijie/lsj/mmskeleton/work_dir/recognition/mff/kinetics-skeleton/epoch_45.pth
