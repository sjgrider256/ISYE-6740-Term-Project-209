# ISYE-6740-Term-Project-209
### Individual tree species classification using hyperspectral NEON data
Guide: https://www.neonscience.org/resources/learning-hub/tutorials/refl-classification-pyxarray

This project explores the use of machine learning for forest species classification using data from the National Ecological Observatory Network (NEON). Field-observed tree species from NEON's Terrestrial Observation System (TOS) are combined with hyperspectral imagery and LiDAR from the Airborne Observation Platform (AOP) to characterize individual canopy trees.

The workflow includes spatially matching field observations to tree crowns, extracting hyperspectral and LiDAR features, reducing spectral dimensionality with PCA, and classifying tree species using SVM, Random Forest, and neural-network models. Performance will be evaluated using spatially grouped validation and a held-out forest plot to test how well the models generalize to unseen areas. Data extraction and preparation methodologies were adopted from the following studies.

### Literature: 
https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002700

https://www.sciencedirect.com/science/article/pii/S0034425722003704
