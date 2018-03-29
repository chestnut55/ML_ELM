import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def obesity_data():
    abundance = 'abundance_obesity.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0, dtype=unicode)
    f = f.T
    f = f.loc[(f != 0).any(axis=1)]
    f.set_index('sampleID', inplace=True)

    l = f['disease'].values
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    # normalize make the elm work better
    f = (f - f.min()) / (f.max() - f.min())


    return train_test_split(f, one_hot_encoded, test_size=0.2, random_state=42)  # Can change to 0.2

def cirrhosis_data():
    abundance = 'abundance_cirrhosis.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0, dtype=unicode)
    f = f.T
    f = f.loc[(f != 0).any(axis=1)]
    f.set_index('sampleID', inplace=True)

    l = f['disease'].values
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    # normalize make the elm work better
    f = (f - f.min()) / (f.max() - f.min())


    return train_test_split(f, one_hot_encoded, test_size=0.2, random_state=42)  # Can change to 0.2

def obesity_gene_marker_data():
    marker_presence = 'marker_presence_obesity.txt'
    f = pd.read_csv(marker_presence, sep='\t', header=None, index_col=0, dtype=unicode)
    f = f.T
    f.set_index('sampleID', inplace=True)


    l = f['disease'].values
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    feature_identifier = "gi|"
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    # normalize make the elm work better
    f = (f - f.min()) / (f.max() - f.min())

    return train_test_split(
        f, one_hot_encoded, test_size=0.2, random_state=42)  # Can change to 0.2


def hmp_hmpii_data():
    abundance = 'abundance_hmp-hmpii-ibd-cirrhosis-t2d_long-t2d_short-WT2D-colorectal--bodysite-stool--disease-n-t2d.txt'
    # abundance = 'abundance_hmp-hmpii-ibd-cirrhosis-t2d_long-t2d_short-colorectal--bodysite-stool.txt'
    f = pd.read_csv(abundance, sep='\t', header=None, index_col=0, dtype=unicode)
    f = f.T
    f = f.loc[(f != 0).any(axis=1)]
    f.set_index('sampleID', inplace=True)

    l = f['disease']
    l = l.replace(to_replace=['small_adenoma', 'large_adenoma'], value=['n', 'n'])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)

    feature_identifier = 'k__'
    feat = [s for s in f.columns if sum([s2 in s for s2 in feature_identifier.split(':')]) > 0]
    f = f.loc[:, feat].astype('float')
    # normalize make the elm work better
    # f = (f - f.min()) / (f.max() - f.min())
    # f = f.div(f.sum(axis=1), axis=0)
    # f -= np.mean(f, axis=0)
    # f /= np.std(f, axis=0)

    return train_test_split(f, one_hot_encoded, test_size=0.1, random_state=42)  # Can change to 0.2


if __name__ == '__main__':
    col1 = hmp_hmpii_data()
    col2 = obesity_data()
    print(len(col2-col1))