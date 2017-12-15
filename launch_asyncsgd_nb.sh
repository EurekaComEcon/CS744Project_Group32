#!/bin/bash

rm serverlog_nb-*
rm asynclog_nb-*

source tfdefs.sh
start_cluster startserver.py
# start multiple clients
echo "running main task"
#nohup python asyncsgd.py --task_index=0 &
nohup python asyncsgd_nb.py --task_index=0 > asynclog_nb-0.out 2>&1&
sleep 10 # wait for variable to be initialized
echo "running other tasks"
nohup python asyncsgd_nb.py --task_index=1 > asynclog_nb-1.out 2>&1&
nohup python asyncsgd_nb.py --task_index=2 > asynclog_nb-2.out 2>&1&
nohup python asyncsgd_nb.py --task_index=3 > asynclog_nb-3.out 2>&1&
nohup python asyncsgd_nb.py --task_index=4 > asynclog_nb-4.out 2>&1&
echo "all tasks are funcitional"
