# Replication code for 'Probabilistic leak localization in water distribution networks using a hybrid data-driven and model-based approach'

### Author
Ganjour Mazaev

### Methods Used
* Machine learning
* Data visualization
* Hydraulic simulation of water distribution networks

### Technologies 
* Python
* Jupyter
* numpy, pandas
* scikit-learn
* matplotlib, plotly
* EPANET, wntr

### Getting started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. The data folder (called /data in the root folder of this repo) is stored on https://doi.org/10.5281/zenodo.6637751, and needs to be copied here.
3. Run ```pip install -e .``` in the root folder of this repository when using it for the first time. This allows src to be imported, which is necessary for most of the code in this project.
4. Install the requirements using ```pip install -r requirements.txt```.
5. Different sets of hydraulic leak scenarios were simulated, which are called a 'HPC run' in this project. The same applies for local leak-free scenarios, which are called 'local runs'. The meaning of these runs is explained [here](docs).
6. All notebooks for visualizing the data, and training/testing the leak localisation models can be found [here](notebooks/analysis). 
7. Use ```pytest``` to run the unit tests in [tests](tests).

## Contact
* ganjour.mazaev@ugent.be, sofie.vanhoecke@ugent.be
