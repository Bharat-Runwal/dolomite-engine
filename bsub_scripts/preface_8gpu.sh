#!/bin/bash

# Default environment variables
# LSB_JOBID :: Job number

# Custom environment variables
export job_name=job1

bsub \
        -q standard \
        -J $job_name \
        -gpu \"num=8/task:mode=exclusive_process\" \
        -n 1 \
        -M 16G \
        -W 01:00 \
        -o $HOME/%J.stdout \
        -e $HOME/%J.stderr \
        < $1

exit 0