import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split


def obesity_data():
    abundance = 'abundance_obesity.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0, dtype=unicode)
    f = f.T
    f.set_index('sampleID', inplace=True)

    # define = '1:disease:obesity'
    # d = pd.DataFrame([s.split(':') for s in define.split(',')])
    # l = pd.DataFrame([0] * len(f))
    # for i in range(len(d)):
    #     tmp = (f[d.iloc[i, 1]].isin(d.iloc[i, 2:])).tolist()
    #     l[tmp] = d.iloc[i, 0]
    #
    # l = l.ix[:,0]
    l = f['disease']

    encoder = LabelEncoder()
    l = pd.Series(encoder.fit_transform(l),
                  index=l.index, name=l.name)

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    # normalize make the elm work better
    f = (f - f.min()) / (f.max() - f.min())

    return train_test_split(
        f, l, test_size=0.2, random_state=42)  # Can change to 0.2

def obesity_gene_marker_data():
    marker_presence = 'marker_presence_obesity.txt'
    f = pd.read_csv(marker_presence, sep='\t', header=None, index_col=0, dtype=unicode)
    f = f.T
    f.set_index('sampleID', inplace=True)

    # define = '1:disease:obesity'
    # d = pd.DataFrame([s.split(':') for s in define.split(',')])
    # l = pd.DataFrame([0] * len(f))
    # for i in range(len(d)):
    #     tmp = (f[d.iloc[i, 1]].isin(d.iloc[i, 2:])).tolist()
    #     l[tmp] = d.iloc[i, 0]
    #
    # l = l.ix[:,0]
    l = f['disease']

    encoder = LabelEncoder()
    l = pd.Series(encoder.fit_transform(l),
                  index=l.index, name=l.name)

    feature_identifier = "gi|"
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    # normalize make the elm work better
    f = (f - f.min()) / (f.max() - f.min())

    return train_test_split(
        f, l, test_size=0.2, random_state=42)  # Can change to 0.2



# only take the normal and cancer data, exclude adenoma data
def colorectal_adenoma_cancer_data():
    # 335 OTUs in total and 490 samples
    otuinfile = 'glne007.final.an.unique_list.0.03.subsample.0.03.filter.shared'
    mapfile = 'metadata.tsv'
    disease_col = 'dx'

    # Data reading
    data = pd.read_table(otuinfile, sep='\t', index_col=1)
    filtered_data = data.dropna(axis='columns', how='all')
    X = filtered_data.drop(['label', 'numOtus'], axis=1)
    metadata = pd.read_table(mapfile, sep='\t', index_col=0)
    y = metadata[disease_col]
    # Inner join metadata and OTU
    merge = pd.concat([X, y], axis=1, join='inner')
    # Filter adenoma
    merge = merge.loc[merge[disease_col].isin(['normal', 'cancer'])]

    y = merge[disease_col]
    X = merge.drop([disease_col], axis=1)

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y), index=y.index, name=y.name)
    return train_test_split(X, y, test_size=0.1, random_state=42)  # Can change to 0.2


# include adenoma data and consider that as normal
def colorectal_cancer_data():
    # Data reading
    otuinfile = 'glne007.final.an.unique_list.0.03.subsample.0.03.filter.shared'
    mapfile = 'metadata.tsv'
    disease_col = 'dx'

    data = pd.read_table(otuinfile, sep='\t', index_col=1)
    filtered_data = data.dropna(axis='columns', how='all')
    X = filtered_data.drop(['label', 'numOtus'], axis=1)
    metadata = pd.read_table(mapfile, sep='\t', index_col=0)
    y = metadata[disease_col]
    ## Merge adenoma and normal in one-category called no-cancer, so we have binary classification
    y = y.replace(to_replace=['normal', 'adenoma'], value=['no-cancer', 'no-cancer'])

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y),
                  index=y.index, name=y.name)
    return train_test_split(X, y, test_size=0.15, random_state=42)

def t2d_data():
    abundance = 'T2D/KEGG_StageII_relative_abun.txt'
    mapfile = 'T2D/stageII.csv'
    disease_col = 'Diabetic'

    # Data reading
    data = pd.read_csv(abundance, sep='\t', header=None, dtype=unicode)
    # first column
    kegg_feat = data.ix[:, 0]
    kegg_feat = kegg_feat[1:]
    # first row
    sampleID = data.ix[0, :]
    sampleID = sampleID[1:]
    # drop the first column
    data.drop(data.columns[[0]], axis=1, inplace=True)
    # drop the first row
    data = data.iloc[1:]
    data = data.T
    data.columns = kegg_feat
    data.set_index(sampleID, inplace=True)
    # filter the zero columns
    data = data.astype('float')
    data = data.loc[:, (data != 0).any(axis=0)]
    # normalize make elm method performance bad
    #data = (data - data.min()) / (data.max() - data.min())
    metadata = pd.read_csv(mapfile, index_col=0)
    y = metadata[disease_col]
    ## Merge adenoma and normal in one-category called no-cancer, so we have binary classification
    # y = y.replace(to_replace=['normal','adenoma'], value=['no-cancer','no-cancer'])

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y),
                  index=y.index, name=y.name)

    return train_test_split(data, y, test_size=0.2, random_state=42)  # Can change to 0.2
