@echo off
echo Installing packages...
pip install pandas
conda install scipy
pip install scikit-image
pip install shapely
pip install simpleitk
pip install rt_utils
pip install git+https://github.com/dicompyler/dicompyler-core.git
conda install matplotlib
conda install setuptools_scm
echo Installation complete!