#!/bin/bash

bsub -Is -n 1 -gpu \"num=2/task:mode=exclusive_process\" /bin/bash
