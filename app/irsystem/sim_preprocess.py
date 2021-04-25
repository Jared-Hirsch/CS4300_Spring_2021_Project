import pandas as pd
import pickle
import unidecode # pylint: disable=import-error
from collections import Counter, defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
# from nltk.corpus import stopwords # pylint: disable=import-error

# stopwords = set(stopwords.words('english')) #can add additional words to ignore
# stopwords = pickle.load(open('stopwords.pkl', 'rb'))


# def set_stopwords(path):
#     global stopwords
#     stopwords = pickle.load(open(path, 'rb'))

# AF_COLS = ['acousticness', 'danceability',
#        'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
#        'speechiness', 'tempo', 'time_signature', 'valence'] #features used in similarity computations

AF_COLS = ['acousticness', 'danceability',
       'energy', 'instrumentalness', 'liveness', 'loudness',
       'speechiness', 'tempo', 'valence']


def make_inv_idx(lyrics_dict, remove_stopwords):
    """
    @params: 
        lyrics_dict: dict; {uri : Counter(tokenized lyrics)} 
        remove_stopwords: Boolean; True if stopwords should be ignored, False otherwise
    @returns:
        dict; {token: [uri, term frequency in corresponding song]}
    
    - creates inverted index for lyrics of songs in dataset 
    """
    inv_idx = defaultdict(list)
    
    for uri, cnt in lyrics_dict.items():
        for word, val in cnt.items():
            if remove_stopwords:
                if word not in stopwords:
                    inv_idx[word].append((uri, val))
            else:
                inv_idx[word].append((uri, val))
                    
    return inv_idx

def compute_idf(inv_idx, n_docs, min_df_ratio = 0.0, max_df_ratio=1.0):
    """
    @params: 
        inv_idx: dict; inverted index 
        n_docs: int; number of songs in dataset
        min_df_ratio: float; min proportion for which a token can occur to be considered
        max_df_ratio: float; max proportion for which a token can occur to be considered
    @returns:
        dict; {token:idf value}
        dict; {token:index of token}
        dict; {index of token: token}

    - creates inverse document frequemcy dict 
    """
    idf_dict = dict()
    word_to_ix = dict()
    ix_to_word = dict()
    ix = 0
    for word, posting in inv_idx.items():
        df = len(posting)
        if (df/n_docs >= min_df_ratio) and (df/n_docs <= max_df_ratio):
            idf_dict[word] = np.log2(n_docs/(1+df))
            word_to_ix[word] = ix
            ix_to_word[ix] = word
            ix += 1
    return idf_dict, word_to_ix, ix_to_word

def compute_song_norms(inv_idx, idf_dict):
    """
    @params: 
        inv_idx: dict; inverted index 
        idf_dict: dict; inverse document frequency
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
        af_matrix: Numpy array; audio features of songs in dataset (n_songs x n_audio_features)
        scalar: StandardScaler; fitted on dataset
    """
    scaler = StandardScaler()
    af_matrix = scaler.fit_transform(df.loc[:, AF_COLS].to_numpy()) #need to scale data, otherwise all scores are .99
    af_song_norms = np.linalg.norm(af_matrix, axis = 1)
    return af_matrix, af_song_norms, scaler

def precompute_lyric_sim(inv_idx, idf_dict, uri_to_ix, word_to_ix):
    n_songs = len(uri_to_ix)
    n_tokens = len(idf_dict)
    tfidf_matrix = np.zeros((n_songs, n_tokens))
    for token, idf in idf_dict.items():
        j = word_to_ix[token]
        for uri, tf in inv_idx[token]:
            i = uri_to_ix[uri]
            tfidf_matrix[i,j] = tf*idf
    cossim_lyrics = cosine_similarity(tfidf_matrix, dense_output = False)
    return cossim_lyrics

def precompute_af_sim(af_matrix):
    return cosine_similarity(af_matrix, dense_output = False)

def preprocess(dataset_path, df_name, lyrics_name, output_name, uri_colname = 'uri', artist_colname = 'artist', name_colname = 'name', remove_stopwords = True, min_df_ratio = 1, max_df_ratio = 1.0, precompute = True, save = True):
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

    ix_to_uri = dict()
    uri_to_ix = dict()
    uri_to_song = dict()
    for i,row in df.iterrows():
        uri = row[uri_colname]
        ix_to_uri[i] = uri
        uri_to_ix[uri] = i
        uri_to_song[uri] = row.to_dict()

    inv_idx = make_inv_idx(lyrics_dict, remove_stopwords)
    idf_dict, word_to_ix, ix_to_word = compute_idf(inv_idx, n_docs, min_df_ratio, max_df_ratio)
    song_norms_dict = compute_song_norms(inv_idx, idf_dict)

    af_matrix, af_song_norms, scaler = get_af_matrix_data(df, uri_colname)

    objs = dict(zip(['uri_to_song', 'inv_idx', 'idf_dict', 'word_to_ix', 'ix_to_word', 'song_norms_dict', 'ix_to_uri', 'uri_to_ix', 'af_matrix', 'af_song_norms', 'scaler'], \
        [uri_to_song, inv_idx, idf_dict, word_to_ix, ix_to_word, song_norms_dict, ix_to_uri, uri_to_ix, af_matrix, af_song_norms, scaler]))
    
    if precompute:
        cossim_lyrics = precompute_lyric_sim(inv_idx, idf_dict, uri_to_ix, word_to_ix)
        cossim_af = precompute_af_sim(af_matrix)
        cossim_matrices = {'lyrics':cossim_lyrics, 'af':cossim_af}


    if save:
        out_name = output_name + "sim_vars.pkl"
        print("\tSaving variables to: ", out_name)
        pickle.dump(objs, open(dataset_path + out_name, 'wb'))

        if precompute:
            out_name = output_name + "cossim_matrices.pkl"
            print("\tSaving cosine similarity matrices to: ", out_name)
            pickle.dump(cossim_matrices, open(dataset_path + out_name, 'wb'))

    return objs

if __name__ == "__main__":
    path = os.getcwd() + os.path.sep + '..' + os.path.sep + '..' + os.path.sep + 'sample_data' + os.path.sep
    df = "SpotifyAudioFeaturesApril2019.csv"
    lyrics = "top_lyrics_annotations.pkl"
    preprocess(path, df, lyrics, 'top_annotations_', 'track_id', 'artist_name', 'track_name', min_df_ratio = 0.01, max_df_ratio = 0.4)


