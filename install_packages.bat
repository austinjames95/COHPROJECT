@echo off
echo Installing packages...
conda install -y pandas
conda install -y scipy
conda install -y scikit-image
pip install simpleitk
pip install rt_utils
pip install git+https://github.com/dicompyler/dicompyler-core.git
echo Installation complete!