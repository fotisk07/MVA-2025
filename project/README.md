# Review of SoccerCPD Paper

## Getting started

### Using UV

Using UV is the easiest way to run this repository. You just need to install soccercpd in editable mode
```python
uv pip install -e soccercpd
```

Then you can just run the soumission notebooks

### Using a venv
If you don't want to use UV it is strongly advised to use a jupyter notebook. 
This codebase uses python3.8 and R (just like in the original paper).

You should install the necessary requirements, and also the package soccercpd in editable mode

### Note on soccercpd package
Soccercpd is the implementation from [SoccerCPD](https://github.com/hyunsungkim-ds/soccercpd), with slight modifications so that it can be used for our project

### Data and results
We provide reade data in a .ugp and .parquet format. We also provide the results of our analysis in .csv files since generating them on your own can take some time.