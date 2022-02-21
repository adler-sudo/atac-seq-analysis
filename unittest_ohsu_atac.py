#!/usr/bin/env python

# import modules
import unittest
import os
from pandas.testing import assert_frame_equal

import ohsu_atac as oa

import pandas as pd

from ohsu_atac import process_metadata
from ohsu_atac import normalize_matrix


# define globals
TEST_DIR = os.path.dirname('test/')
TEST_METADATA_FILE = os.path.join(TEST_DIR,'test_metadata.txt')
TEST_ATACSEQ_FILE = os.path.join(TEST_DIR,'test_atac.csv')

# build unit tests
class TestEssentials(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_read_metadata(self):
        metadata = oa.read_metadata(TEST_METADATA_FILE)
        metadata.columns = ['type']
        metadata_compare = pd.DataFrame({
            'type': ['excitatory_neuron','excitatory_neuron','inhibitory_neuron','excitatory_neuron','inhibitory_neuron']
        })
        assert_frame_equal(metadata,metadata_compare)

    def test_process_metadata(self):
        metadata = oa.read_metadata(TEST_METADATA_FILE)
        df = pd.read_csv(TEST_ATACSEQ_FILE,index_col=0)
        
        metadata = oa.process_metadata(df,metadata)
        metadata_compare = pd.DataFrame(
            {
                'sample_id':['excitatory_neuron','excitatory_neuron','inhibitory_neuron','excitatory_neuron','inhibitory_neuron'],
                'count':[3, 2, 5, 10, 3],
                'std':[0.957427,1.000000,2.500000,3.785939,1.500000],
                'mean':[0.75,0.50,1.25,2.50,0.75],
                'zero':[2,3,3,2,3]
            },
            index=['GATTCGGTAGTTACGCAAGTCCAA','TGCGGCCTGATCATGAAGCTCGCT','GATTCGGTACCGGAAGCGTTAGAA','TGCGGCCTGCCGGAGCAGTTCAGG','ACGCGACGAATGATGCGATCTATC']
            )
        assert_frame_equal(metadata,metadata_compare)

    def test_filter_matrix(self):
        # define filter and comparisons
        FILTER_THRESHOLD = 4
        FILTERED_DF_COMPARE = pd.DataFrame(
            {
                'GATTCGGTACCGGAAGCGTTAGAA':[5, 0, 0, 0],
                'TGCGGCCTGCCGGAGCAGTTCAGG':[8, 0, 0, 2]
            },
            index=['chr1-3094454-3095231','chr1-3117883-3118383','chr1-3119738-3120238','chr1-3120562-3121062']
        )
        FILTERED_METADATA_COMPARE = pd.DataFrame(
            {
                'sample_id':['inhibitory_neuron','excitatory_neuron'],
                'count':[5, 10],
                'std':[2.500000,3.785939],
                'mean':[1.25,2.50],
                'zero':[3, 2]
            },
            index=['GATTCGGTACCGGAAGCGTTAGAA','TGCGGCCTGCCGGAGCAGTTCAGG']
        )
        
        # read in and process parent files
        metadata = oa.read_metadata(TEST_METADATA_FILE)
        df = pd.read_csv(TEST_ATACSEQ_FILE,index_col=0)
        metadata = oa.process_metadata(df,metadata)

        # perform filter
        filtered_df, filtered_metadata = oa.filter_matrix(df,metadata,filter=FILTER_THRESHOLD)

        # confirm equality
        assert_frame_equal(filtered_df,FILTERED_DF_COMPARE)
        assert_frame_equal(filtered_metadata,FILTERED_METADATA_COMPARE)

    def test_normalize_matrix(self):
        # define comparison
        NORMALIZED_DF_COMPARE = pd.DataFrame()

        # read in parent data and reduce size
        df = pd.read_csv(TEST_ATACSEQ_FILE,index_col=0)
        df = df.iloc[0:2,0:2]

        # normalize
        normalized_df = oa.normalize_matrix(df)
        print(normalized_df)

        # LEFT OFF HERE
        self.assertEqual(1,2)

if __name__ == '__main__':
    unittest.main()