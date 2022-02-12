#!/usr/bin/env python

# import modules
from locale import normalize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px

import pandas as pd
import numpy as np

import argparse


# argparse
def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Dimensionality reduction of ATAC-seq')

    parser.add_argument('--metadata',help='Metadata file')
    parser.add_argument('--atac_counts',help='ATAC-seq counts data')
    parser.add_argument('--filter_threshold',help='Minimum samples with at least 1 read',type=int,default=10)
    parser.add_argument('--n_components',help='Number PCA components.',type=int,default=2)
    args = parser.parse_args()
    return args

# filter
def filter_matrix(df,filter):
    """
    filter out peaks with fewer than 'filter' reads
    """
    # remove peaks for which there are fewer than filter reads
    # TODO: need to adjust this (only want to remove 0s, not lowly expressed genes)
    filtered_df = df.loc[(df != 0).sum(axis=1) > filter,:]
    print(f'{len(df) - len(filtered_df)} peaks removed from dataset')
    return filtered_df

# normalize
def normalize_matrix(df):
    """
    normalize matrix so each sample has equal number of reads
    """
    normalized_df = df * df.sum(axis=0).max() / df.sum(axis=0).values
    return normalized_df

# transformation
def log_transformation(df):
    """
    performs log transformation of dataframe
    """
    log_df = np.log10(df + 1)
    return log_df

# dimensionality reduction
def reduce_dimensions(n_components:int=2,df=None):
    """
    reduce dimensions of atac-seq
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)
    print(pca.explained_variance_ratio_)
    return components

# metadata
def read_metadata(file:str):
    """
    read in metadata
    """
    metadata = pd.read_csv(file,header=None)
    return metadata

# run program
if __name__ == '__main__':

    # arguments
    args = parse_args()
    df = pd.read_csv(args.atac_counts,index_col=0)
    metadata = read_metadata(args.metadata)
    print(metadata.head())

    # normalize and filter data
    df = normalize_matrix(df)
    df = filter_matrix(df,args.filter_threshold)

    # transform data
    df = log_transformation(df)

    # conduct pca
    components = reduce_dimensions(n_components=args.n_components,df=df)





