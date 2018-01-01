# Machine Learning Experiments

## Intallation

This package assumes Keras, tensorflow, h5py, pyplot, pandas and some more
libraries installed in a Python3 virtual environment.

Create it using requirements.txt and pip3.

```
sudo apt-get install python3-pip python3-dev python-virtualenv
```

Then create venv

```
virtualenv --system-site-packages -p python3 ~/tensorflow/
```

Then activate the environment

```
. activate.sh
```

Update pip:
```
easy_install -U pip
```

Use pip3 to install it all.


It should work with requirements.txt, or alternatively 
with manual specification of all packages


```
pip3 install --upgrade -r requirements.txt
```

or


```
pip3 install --upgrade tensorflow keras h5py graphviz pydot pandas
```
