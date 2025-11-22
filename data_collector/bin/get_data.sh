#!/bin/bash
source data_collector/conf/collector.conf

echo "Form ID is set as: ${GDRIVE_FILE_ID}"
echo "Target path is set as: ${path}"

curl -L "https://drive.google.com/uc?export=download&id=${GDRIVE_FILE_ID}" \
  -o "${path}/data.csv"