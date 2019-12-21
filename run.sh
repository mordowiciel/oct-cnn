#!/usr/bin/env python

CONFIG_DIR = '../configs/*'
for config in CONFIG_DIR
do
     echo "Processing config $config ..."
     cat $config
done