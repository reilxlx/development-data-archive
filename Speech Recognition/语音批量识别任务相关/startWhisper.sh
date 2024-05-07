#!/bin/bash
base_port=5001  # 设置基础端口号
for i in {0..1}; do
  port=$(($base_port + $i))  # 为每个GPU计算一个端口号
  CUDA_VISIBLE_DEVICES=$i python runWhisper_v3_multiGPU.py --port $port &
done
