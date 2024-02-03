#! /bin/zsh

requirements="./requirements.txt"
project_dir="../."

cd "${project_dir}"
python3 -m pip install --upgrade pipreqs
python3 -m pipreqs.pipreqs --force --savepath "${requirements}" --encoding utf-8
