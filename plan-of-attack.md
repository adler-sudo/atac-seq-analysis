# Objective

Extract information from an ATAC-seq file. 

# Questions

1. Visualize the cells. 

    a. Do you see a clear separation between cell types? Comment on results.

2. Train classifier to classify astrocyte cells against other cell tyeps. 

    a. Asses performance.

    b. Reasons behind good/bad performance.

3. Suggest analyses to gather biological insights from this data. 

    a. Analyses to distinguish astrocytes from other cell types?

    b. Can we identify upstream biological factors that cause these differences?

# Plan

## Understand the data type

* head file
* read in ATAC data
* read in metadata

## Dimensionality reduction

* use sklearn PCA
    * not sure how data is measured at this point

    * plot different components of the metadata
        * specifically **astrocyte** cells

## Build classifier

* build random forest classifier sklearn
* is there another type of classifier that theresa might find interesting?

## Additional analyses

* Can pull info from our random forest classifier
* Upstream biological factors
    * is there information that exists in the metadata?
    * can we construct some kind of linear model?
