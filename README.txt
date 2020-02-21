This file presents the instructions in order to reproduce the results
presented in our homework. Feel free to contact us at our email adress :
 - morgane_hoffmann@ensae.fr
 - cesar.roaldes@ensae.fr


###########
# SUMMARY #
###########

The instructions has to be done in the following order :
I   - BAYESIAN FOREST IMPLEMENTATION
II  - RUN R SCRIPT part1.R
III - RUN PYTHON SCRIPT part2.py AND part3.py

The R script generates the data and the OLS estimation reused in the
python scripts.


##################################
# BAYESIAN FOREST IMPLEMENTATION #
##################################

We used python to implement the bayesian forest. You need to have Anaconda 
distribution installed to easily get the virtual environment (venv) we 
used on our computers and modify the ensemble/ folder of the sklearn 
package. Here are the instructions to do it.

Choose the .yml file corresponding to your OS in the directory env/ and
install the venv with the following command in console :

conda env create -f env/environment_OS.yml

with the corresponding OS. This manipulation will install a venv
named bf. To activate the env from the console, use :

conda activate bf

Once the venv load, you have to target the folder of your venv. You can
find it by typing (on macOS and Linux distributions) :

which python

that will return somthing like "/anaconda3/envs/bf/bin/python" if everything
worked fine until here. Hence, the corresponding location of the venv
is "/anaconda3/envs/bf/". Then open the folder location of ensemble/ which is,
from the venv location in "bf_path/lib/python3.7/site-packages/sklearn". In
our example, we would type :

cd /anaconda3/envs/bf/lib/python3.7/site-packages/sklearn

Open the folder in graphical user interface by typing :
 - Ubuntu : nautilus .
 - MacOS : open .

From here, delete the ensemble/ folder from the sklearn/ folder and remplace
it with the file contained in our file corresponding to your distribution
(ensemble_ubuntu or ensemble_macOS) and rename this folder "ensemble".

Those files contain the modified script "forest.py" including the bayesian 
forest. Then, you should be able to use the bayesian forest from our script
ever since the venv bf is activated.
