# Uncertainty Quantification for AMPL Point Predictions

## Overview
This repository contains the research work on applying machine learning models to predict the binding or inhibition activity of NimA-related kinases. The project is part of a collaboration with the Accelerating Therapeutics for Opportunities in Medicine (ATOM) Consortium. It focuses on addressing challenges posed by imbalanced and scarce experimental data through uncertainty quantification techniques.

## Installation
To set up this project locally, follow the steps below:
```bash
git clone https://github.com/TsukiTiger/NEK9_Final.git
cd NEK9_Final
pip install -r requirements.txt
```

# To activate this environment, use
#
#     $ conda activate nek2_final
#
# To deactivate an active environment, use
#
#     $ conda deactivate


conda env create -f environment.yml 
conda activate nek2_final
python -m ipykernel install --user --name nek2_final 
 
## Usage
To run the main analysis notebook, use the following command:
```bash
code .
cd notebooks
```
Ensure you have the necessary data files in the appropriate directories as expected by the scripts.

## Data
Due to confidentiality agreements, raw data files are not included in this repository. Data used in this project are part of the ATOM Consortium's private datasets. Please ensure you have the correct permissions to access the data.

## Models
This project includes several machine learning models aimed at classifying drug compounds. The models are saved in the `models/` directory after training and can be loaded for further analysis or prediction as follows:
```python
import joblib
model = joblib.load('models/model_name.pkl')
```

## Contributing
We welcome contributions from the community. Please fork the repository and submit a pull request with your proposed changes. For major changes, please open an issue first to discuss what you would like to change.

## Contact
- **Chongye Feng** - chongyef@gmail.com
- **Ya Ju Fan, PhD** - Mentor's Email Here
- **Amanda Paulson, PhD** - Mentor's Email Here
