# Document for the cluster algorithms


## Table of Contents
- [Importing Data and Preprocessing](#Title-One)
- [Title Two](#Title-Two)
- [Title Three](#Title-Three)

## Importing Data and Preprocessing
Before importing data, it is necessary to choose the data processing method. If the data processing method chosen is to handle missing values (NAN), first check the columns. If the number of missing values in a column is greater than 20% of the total number of columns, delete that column directly. Then, proceed to check the rows. If there are rows with missing data, delete those rows.

For clustering high-dimensional data, the scale of the data has a significant impact on clustering. Consider standardizing each column, i.e., each dimension, of the data. Standardization places the data on the same scale.

Z-score Standardization:

$$x_{i j}{ }^{\prime}=\frac{x_{i j}- {mean}\left(x_j\right)}{ {std}\left(x_j\right)}$$

Mapminmax Standardization:

$$x_{i j}^{\prime}=\frac{x_{i j}-\min \left(x_{i j}\right)}{\max \left(x_j\right)-\min \left(x_j\right)}$$

## Title Two
This is the content of title two.

## Title Three
This is the content of title three.



