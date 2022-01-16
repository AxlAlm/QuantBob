#!/bin/bash

# python version
python_path="/home/axl/.cache/pypoetry/virtualenvs/quantbob-Vnca5bs1-py3.8/bin/python3.8"

# path to script
script_path="/home/axl/repos/QuantBob/run.py"

# path to config file
config_file="/home/axl/repos/QuantBob/model_config.toml"

# run the stuff
echo $python_path $script_path -c $config_file -d
$python_path $script_path -c $config_file -d