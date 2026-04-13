#!/bin/bash

bsub -Is -n 1 -gpu \"num=2\" /bin/bash
