# Document for the combination of clustering algorithms

## Table of Contents
- [Importing Data and Preprocessing](#importing-data-and-preprocessing)
- [Dimensionality Reduction](#dimensionality-reduction)
  - [Principal component analysis](#principal-component-analysis)
  - [Non-negative matrix factorization](#non-negative-matrix-factorization)
  - [Singular value decomposition](#singular-value-decomposition)
  - [Stochastic neighbor embedding](#stochastic-neighbor-embedding)
  - [t-distributed stochastic neighbor embedding](#t-distributed-stochastic-neighbor-embedding)
  - [Kernel principal component analysis](#kernel-principal-component-analysis)


## Importing Data and Preprocessing
Before importing data, it is necessary to choose the data processing method. If the data processing method chosen is to handle missing values (NAN), first check the columns. If the number of missing values in a column is greater than 20% of the total number of columns, delete that column directly. Then, proceed to check the rows. If there are rows with missing data, delete those rows.

For clustering high-dimensional data, the scale of the data has a significant impact on clustering. Consider standardizing each column, i.e., each dimension, of the data. Standardization places the data on the same scale.

Z-score Standardization:

$$x_{i j}{ }^{\prime}=\frac{x_{i j}- {mean}\left(x_j\right)}{ {std}\left(x_j\right)}$$

Mapminmax Standardization:

$$x_{i j}^{\prime}=\frac{x_{i j}-\min \left(x_{i j}\right)}{\max \left(x_j\right)-\min \left(x_j\right)}$$

## Dimensionality Reduction
After importing the data, the next module involves dimensionality reduction. If you only wish to perform clustering without dimensionality reduction, you can opt not to reduce dimensions. Otherwise, you may choose other dimensionality reduction methods before proceeding with the reduction. The toolbox defaults to considering data columns as features and performs dimensionality reduction on these feature columns.

If high-dimensional data visualization is selected, MATLAB's built-in tsne function is used for t-SNE dimensionality reduction. This function reduces the high-dimensional data to 2 dimensions and displays it in a two-dimensional space.

This section outlines the process of dimensionality reduction after data import, allowing users to choose whether to perform dimensionality reduction and, if so, which method to use. Additionally, it provides information on the default dimensionality reduction approach and the specific method used for high-dimensional data visualization using t-SNE. If you need further assistance, feel free to ask!


### Principal component analysis (PCA)
Link: https://en.wikipedia.org/wiki/Principal_component_analysis

### Non-negative matrix factorization (NNMF)
Link: https://en.wikipedia.org/wiki/Non-negative_matrix_factorization

### Singular value decomposition (SVD)
Link: https://en.wikipedia.org/wiki/Singular_value_decomposition

### Stochastic neighbor embedding （SNE）
Link: https://papers.nips.cc/paper_files/paper/2002/hash/6150ccc6069bea6b5716254057a194ef-Abstract.html

### t-distributed stochastic neighbor embedding (t-SNE)
Link: https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding

### Kernel principal component analysis  (kernel PCA)
Link: https://en.wikipedia.org/wiki/Kernel_principal_component_analysis







