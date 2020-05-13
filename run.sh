#!/usr/bin/env bash
for data_file in dev test train; do
export data_file
sbatch --partition titanx-long --gres=gpu:1 --mem 100GB \
-o mask_process_${data_file}.stdout.txt \
-e mask_process_${data_file}.stderr.txt \
--job-name=mask_process job.sbatch
done