#!/usr/bin/env python

# import modules
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import PowerTransformer
import umap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import argparse


# argparse
def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Dimensionality reduction of ATAC-seq')

    parser.add_argument('--metadata',help='Metadata file')
    parser.add_argument('--atac_counts',help='ATAC-seq counts data')
    parser.add_argument('--filter_threshold',help='Minimum counts per sample.',type=int,default=0)
    parser.add_argument('--n_components',help='Number PCA components.',type=int,default=50)

    parser.add_argument('--umap_first_pc',help='First principal component to use in UMAP.',type=int,default=2)
    parser.add_argument('--umap_last_pc',help='Last principal component to use in UMAP.',type=int,default=12)
    parser.add_argument('--svc_first_pc',help='First principal component to use in SVC.',type=int,default=0)
    parser.add_argument('--svc_last_pc',help='Last principal component to use in SVC.',type=int,default=12)

    parser.add_argument('--umap_color',help='Metadata column to color UMAP by.',type=str,default='sample_id')
    parser.add_argument('--dim_reduce_image',help='Name of dimension reduced image.',type=str,default='dim-reduce.png')
    parser.add_argument('--svc_image',help='Name of SVC heatmap image.',type=str,default='svc-image.png')

    parser.add_argument('--test_size',help='Test size of split for training data.',type=float,default=0.3)
    
    args = parser.parse_args()
    return args

# process metadata
def process_metadata(df,metadata):
    """
    change row names and add summary columns to metadata
    """
    # set index equal to sample ids
    metadata.index = df.columns
    metadata.columns = ['sample_id']

    # include summary files    
    metadata['count'] = df.sum(axis=0).values
    metadata['std'] = df.std(axis=0).values
    metadata['mean'] = df.mean(axis=0).values
    metadata['zero'] = (df==0).sum()

    return metadata

# filter
def filter_matrix(df,metadata,filter):
    """
    filter out peaks with fewer than 'filter' reads
    """
    # remove peaks for which there are fewer than filter reads
    # TODO: need to adjust this (only want to remove 0s, not lowly expressed genes)
    count_mask = df.sum(axis=0) > filter

    filtered_df = df.loc[:,count_mask]
    filtered_metadata = metadata.loc[count_mask,:]
    print(f'{df.shape[1] - filtered_df.shape[1]} samples removed from dataset')

    return filtered_df, filtered_metadata

# normalize
def normalize_matrix(df):
    """
    normalize matrix so each sample has equal number of reads
    """
    # normalize
    normalized_df = df * df.sum(axis=0).median() / df.sum(axis=0).values
    
    return normalized_df

# transformation
def log_transformation(df):
    """
    performs log transformation of dataframe
    """
    log_df = np.log10(df + 1)
    return log_df

# power transform
def boxcox_transformation(df):
    """
    perform boxcox transformation
    """
    trans_df = PowerTransformer(method="box-cox").fit_transform(df+1)
    return trans_df

# dimensionality reduction
def reduce_dimensions(df,n_components):
    """
    reduce dimensions of atac-seq
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df)
    return components

# UMAP
def generate_umap(df,random_state=0,n_epochs=30000):
    """
    generate components of umap
    """
    print(df.shape)
    reducer = umap.UMAP(random_state=random_state,n_epochs=n_epochs)
    components = reducer.fit_transform(df)

    return components

# metadata
def read_metadata(file:str):
    """
    read in metadata
    """
    metadata = pd.read_csv(file,header=None)
    return metadata

# construct image
def construct_image(x,y,labels,file:str):
    """
    constructs scatter plot of x by y and saves to file
    """
    p = sns.scatterplot(x=x,y=y,hue=labels,size=1)
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    plt.title('UMAP cluster of sc-ATAC-seq data')
    plt.savefig(file,bbox_inches='tight')
    return None

# perform classification
def construct_svc(X_train,y_train,random_state=0):
    """
    generate svc to classify cell type given ATAC-seq data
    """
    svc = SVC(random_state=0)
    svc.fit(X_train,y_train)
    return svc

# evaluate svc
def evaluate_svc(svc,X_test,y_test):
    """
    returns accuracy of svc
    """
    accuracy = svc.score(X_test, y_test)
    return accuracy

# confusion matrix
def generate_confusion_matrix(svc,X_test,y_test,cmap='Blues',file='conf-matrix.png'):
    """
    generates and saves confusion matrix
    """
    plot_confusion_matrix(svc,X_test, y_test, cmap=cmap)
    plt.title('True label vs predicted label for SVC')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(file)

# run program
if __name__ == '__main__':

    # arguments
    args = parse_args()
    df = pd.read_csv(args.atac_counts,index_col=0)
    metadata = read_metadata(args.metadata)

    # set index and summary columns
    metdaata = process_metadata(df,metadata)

    # normalize and filter data
    df, metadata = filter_matrix(df,metadata,args.filter_threshold)
    df = normalize_matrix(df)
    
    # transform data
    df = boxcox_transformation(df)

    # conduct pca
    components = reduce_dimensions(
        n_components=args.n_components,
        df=df.transpose())

    # execute UMAP
    print(components.shape)
    umap_components = generate_umap(components[:,args.umap_first_pc:args.umap_last_pc])

    # generate plot
    construct_image(
        x=umap_components[:,0],
        y=umap_components[:,1],
        labels=metadata[args.umap_color].values,
        file=args.dim_reduce_image)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        components[:,args.svc_first_pc:args.svc_last_pc], 
        metadata['sample_id'].values, 
        test_size=args.test_size,
        random_state=0,
        stratify=metadata['sample_id'].values)

    # construct and evaluate classifier
    svc = construct_svc(X_train,y_train)
    accuracy_train = evaluate_svc(svc, X_train, y_train) * 100
    accuracy_test = evaluate_svc(svc, X_test, y_test) * 100
    print(f'SVC train set accuracy: {accuracy_train}')
    print(f'SVC test set accuracy: {accuracy_test}')

    # generate confusion matrix
    generate_confusion_matrix(svc, X_test, y_test, file=args.svc_image)
