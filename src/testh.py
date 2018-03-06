import numpy as np
import pandas as pd

if __name__ == '__main__':
    ###################### Stage I #########################
    stageI_abundance = 'T2D/KEGG_StageI_relative_abun'
    stageI_meta_data = 'T2D/stageI.csv'
    stageI_meta_data = pd.read_csv(stageI_meta_data, index_col=0)
    stageI_y = stageI_meta_data['Diabetic (Y or N)']

    stageI_data = pd.read_csv(stageI_abundance, sep='\t', header=None, dtype=unicode)
    stageI_kegg_feat = stageI_data.ix[:, 0]
    stageI_kegg_feat = stageI_kegg_feat[1:]

    stageI_sampleID = stageI_data.ix[0, :]
    stageI_sampleID = stageI_sampleID[1:]

    # drop the first column and first row
    stageI_data.drop(stageI_data.columns[[0]], axis=1, inplace=True)
    stageI_data = stageI_data.iloc[1:]

    stageI_data = stageI_data.T
    stageI_data.columns = stageI_kegg_feat
    stageI_data.set_index(stageI_sampleID, inplace=True)
    # filter the zero columns
    stageI_data = stageI_data.astype('float')
    stageI_data = stageI_data.loc[:, (stageI_data != 0).any(axis=0)]

    ###################### Stage II #########################
    stageII_abundance = 'T2D/KEGG_StageII_relative_abun.txt'
    stageII_meta_data = 'T2D/stageII.csv'

    # Data reading
    stageII_data = pd.read_csv(stageII_abundance, sep='\t', header=None, dtype=unicode)
    # first column
    stageII_kegg_feat = stageII_data.ix[:, 0]
    stageII_kegg_feat = stageII_kegg_feat[1:]
    # first row
    stageII_sampleID = stageII_data.ix[0, :]
    stageII_sampleID = stageII_sampleID[1:]
    # drop the first column
    stageII_data.drop(stageII_data.columns[[0]], axis=1, inplace=True)
    # drop the first row
    stageII_data = stageII_data.iloc[1:]
    stageII_data = stageII_data.T
    stageII_data.columns = stageII_kegg_feat
    stageII_data.set_index(stageII_sampleID, inplace=True)
    # filter the zero columns
    stageII_data = stageII_data.astype('float')
    stageII_data = stageII_data.loc[:, (stageII_data != 0).any(axis=0)]
    # normalize make elm method performance bad
    #data = (data - data.min()) / (data.max() - data.min())
    stagteII_y = pd.read_csv(stageII_meta_data, index_col=0)['Diabetic']
    #############################################################

    aa = set(list(stageII_data)) - set(list(stageI_data))
    bb = set(list(stageI_data)) - set(list(stageII_data))

    print aa
    print bb