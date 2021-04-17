import pandas as pd
import pickle
import unidecode # pylint: disable=import-error
from collections import Counter, defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
# from nltk.corpus import stopwords # pylint: disable=import-error

# stopwords = set(stopwords.words('english')) #can add additional words to ignore
stopwords = pickle.load(open('stopwords.pkl', 'rb'))



AF_COLS = ['acousticness', 'danceability',
       'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'time_signature', 'valence'] #features used in similarity computations

def make_inv_idx(lyrics_dict, remove_stopwords):
    """
    @params: 
        lyrics_dict: dict; {uri : Counter(tokenized lyrics)} 
        remove_stopwords: Boolean; True if stopwords should be ignored, False otherwise
    @returns:
        dict; {token: [uri, term frequency in corresponding song]}
        dict; {token: integer index}
    
    - creates inverted index for lyrics of songs in dataset 
    """
    word_to_ix = dict()
    inv_idx = defaultdict(list)
    word_ix = 0
    
    for uri, cnt in lyrics_dict.items():
        for word, val in cnt.items():
            if remove_stopwords:
                if word not in stopwords:
                    inv_idx[word].append((uri, val))
                    if word not in word_to_ix:
                        word_to_ix[word] = word_ix
                        word_ix += 1
            else:
                inv_idx[word].append((uri, val))
                if word not in word_to_ix:
                    word_to_ix[word] = word_ix
                    word_ix += 1
                    
    return inv_idx, word_to_ix

def compute_idf(inv_idx, n_docs, min_df_ratio = 0.0, max_df_ratio=1.0):
    """
    @params: 
        inv_idx: dict; inverted index 
        n_docs: int; number of songs in dataset
        min_df: int; minimum term frequency for token to be considered
        max_df_ratio: float; threshold for proportion of times a token can occur
    @returns:
        dict; {token:idf value}
    
    - creates inverse document frequemcy dict 
    """
    idf_dict = dict()
    for word, posting in inv_idx.items():
        df = len(posting)
        if (df/n_docs >= min_df_ratio) and (df/n_docs <= max_df_ratio):
            idf_dict[word] = np.log2(n_docs/(1+df))
    return idf_dict

def compute_song_norms(inv_idx, idf_dict):
    """
    @params: 
        inv_idx: dict; inverted index 
        idf_dic: dict; inverse document frequency
    @returns:
        dict; {uri:norm of song's tfidf vector}
    """
    song_norms_dict = dict()
    for word, postings in inv_idx.items():
        for uri, tf in postings: 
            song_norms_dict[uri] = song_norms_dict.get(uri,0) + (tf*idf_dict.get(word, 0))**2
    return {k:np.sqrt(v) for k,v in song_norms_dict.items()}

def get_af_matrix_data(df, uri_colname):
    """
    @params: 
        df: DataFrame; dataframe of song's audio features 
        uri_colname: String; name of column containing song's URI
    @returns:
        af_ix_to_uri: dict; {index of song in dataframe : song's uri}
        af_uri_to_ix: dict; {uri : index of song in dataframe}
        af_matrix: Numpy array; audio features of songs in dataset (n_songs x n_audio_features)
        scalar: StandardScaler; fitted on dataset
    """
    af_ix_to_uri = {i:row[uri_colname] for i, row in df.iterrows()}
    af_uri_to_ix = {uri:i for i,uri in af_ix_to_uri.items()}
    scaler = StandardScaler()
    af_matrix = scaler.fit_transform(df.loc[:, AF_COLS].to_numpy()) #need to scale data, otherwise all scores are .99
    af_song_norms = np.linalg.norm(af_matrix, axis = 1)
    return af_ix_to_uri, af_uri_to_ix, af_matrix, af_song_norms, scaler

def preprocess(dataset_path, df_name, lyrics_name, output_name, uri_colname = 'uri', artist_colname = 'artist', name_colname = 'name', remove_stopwords = True, min_df_ratio = 1, max_df_ratio = 1.0, save = True):
    """
    @params: 
        dataset_path: String; directory in which dataset is stored 
        df_name: String; name of file containing dataset
        lyrics_name: String; name of file containing lyrics
        save: Boolean; if True, saves variables to specified directory as 'sim_vars.pkl'
    @returns:
       obj: dict of variables
    """
    df = pd.read_csv(dataset_path + df_name)
    lyrics_dict = pickle.load(open(dataset_path + lyrics_name, 'rb'))
    n_docs = len(lyrics_dict)
    df = df.loc[df.track_id.isin(lyrics_dict)].reset_index(drop = True) #only use songs with retrieved lyrics
    uri_to_song = {row[uri_colname]:row.to_dict() for _, row in df.iterrows()}

    inv_idx, word_to_ix = make_inv_idx(lyrics_dict, remove_stopwords)
    idf_dict = compute_idf(inv_idx, n_docs, min_df_ratio, max_df_ratio)
    song_norms_dict = compute_song_norms(inv_idx, idf_dict)

    ix_to_uri, uri_to_ix, af_matrix, af_song_norms, scaler = get_af_matrix_data(df, uri_colname)

    objs = dict(zip(['uri_to_song', 'inv_idx', 'word_to_ix', 'idf_dict', 'song_norms_dict', 'ix_to_uri', 'uri_to_ix', 'af_matrix', 'af_song_norms', 'scaler'], \
        [uri_to_song, inv_idx, word_to_ix, idf_dict, song_norms_dict, ix_to_uri, uri_to_ix, af_matrix, af_song_norms, scaler]))
    
    if save:
        out_name = output_name + "sim_vars.pkl"
        print("Saving variables to: ", out_name)
        pickle.dump(objs, open(dataset_path + out_name, 'wb'))
    
    return objs

if __name__ == "__main__":
    path = r"C:\Users\chris\Documents\GitHub\cs4300sp2021-rad338-jsh328-rpp62-cmc447\sample_data/"
    df = "SpotifyAudioFeaturesApril2019.csv"
    lyrics = "top_lyrics_annotations.pkl"
    preprocess(path, df, lyrics, 'top_annotations_', 'track_id', 'artist_name', 'track_name', min_df_ratio = 0.01, max_df_ratio = 0.4)


