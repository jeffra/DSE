#!/bin/bash

deepspeed --num_gpus=2 cifar10_tutorial.py --deepspeed_config ds_config.json
