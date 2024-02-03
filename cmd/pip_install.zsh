#! /bin/zsh
requirements="../requirements_new.txt"

python3 -m pip install --upgrade pip
python3 -m pip install -r "${requirements}"
