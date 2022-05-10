# rank_of_medicine_latent_factor_modelling
Using NMF and latent factor modelling to estimate true rank of a medical dataset.
We show that it is possible to fit a lower-rank factor model on a dataset and use this to estimate clinically meaningful data in the dataset. 
For glucose the rank appears to be ~80-120 depending on the task.
The utility of this is in casual inference where we provide support for the use of large-scale PSM. We show that high correlations in the data mean that even if a feature is missing, it is possible to capture it indirectly using latent factor modelling.

Methods:
 - data comes from CUIMC using OMOP formatted CDM
 - We use sparse-NMF to fit the models
 - We evaluate in 3 ways:
    - Reconstruction error in test set using learned variables of train set
    - Masked imputation error in full dataset following masking out 20% outcome variable
    - Regression error using test set latent variables after latent model and regression model fitting in train set

Results:

Glucose
   
 
![image](https://user-images.githubusercontent.com/43360672/167687077-c4224718-a382-45e1-a0d8-6c7869b201ba.png)


HBA1C

  

 
![image](https://user-images.githubusercontent.com/43360672/167687131-2fee0934-da74-44bf-8e70-4c21dd039af7.png)
