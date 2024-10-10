# Causal Infernece Project
### The effect of past military service on working in the public sector in the United States ✪ 
Final project in Technion's Intro to Causal Inference course 
(course number: 097400).


##### Abstract:
This study investigates the influence of a military background on the probability to work in public 
sector employment in the United States. Utilizing data from the American Community Survey (ACS) and 
employing three distinct analytical approaches for ATE calculation, we assess the causal 
relationship between military experience and public sector employment. 
Our findings suggest that a military background increases the likelihood of working in the US public sector, 
particularly at the federal level. 
The analysis reveals that the impact of military service on public sector employment is contingent upon the 
governmental tier, demonstrating a stronger effect at the federal level compared to local and state levels. 
This project underscores the significance of military service as a determinant in the career trajectories of 
veterans and proposes avenues for further research to enrich the comprehension of these dynamics.


##### How to run the code
We managed to run the code in this repository on a virtual machine using the following sequence of commands:

```shell
conda create -n <YOUR_ENV_NAME> python=3.9
conda activate <YOUR_ENV_NAME>
conda install scipy scikit-learn pandas
conda update --all
conda install -c conda-forge libstdcxx-ng
pip install folktables
cd <path/to/project/dir>
python3 experiment.py
```

