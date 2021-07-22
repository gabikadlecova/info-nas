#!/bin/bash

env_dir=$1

if [ -z $env_dir ]; then
  env_dir='.'
fi

python3 -m venv "$env_dir"/pyt
. "$env_dir"/pyt/bin/activate

pip install --upgrade pip
pip install -r "./requirements.txt"
pip install "../nasbench/"
pip install "../NASBench-PyTorch/"
pip install "../arch2vec/"
pip install "."

