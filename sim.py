import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
import unidecode
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
from lyricsgenius import Genius
import re
import time
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import spotipy.util as util
from sp_client import Spotify_Client
from sim_preprocess import AF_COLS
import string

punct = set(string.punctuation)
punct.update({"''", "``", ""})
tokenizer = TreebankWordTokenizer()

stopwords = set(stopwords.words('english'))

path = r'C:\Users\chris\Documents\GitHub\cs4300sp2021-rad338-jsh328-rpp62-cmc447\sample_data/'
vars_dict = pickle.load(open(path + 'sim_vars.pkl', 'rb'))

def strip_name(name):
    """
    @params:
        name: String, track name
    @returns:
        String
    
    - removes extraneous characters in track name, as they may cause issues when querying Spotify/Genius APIs
    """
    for s in ["-", "(", "feat."]:
        ix = name.find(s)
        if s != -1:
            name = name[:ix].strip()
    return name

def match(a, b):
    """
    @params: 
        a: String
        b: String
    @returns:
        Boolean
    
    - basic fuzzy-matching function, used to check whether artists/track names match
    """
    a = unidecode.unidecode(a).lower() #converts special characters to ASCII representation
    b = unidecode.unidecode(b).lower()
    return (a in b) or (b in a)

def retrieve_lyrics(query_artist, query_name, genius):
    """
    @params: 
        query_artist: String
        query_name: String
        genius: Genius object
    @returns:
        Counter of tokenized lyrics or None
    
    - helper function for lyrics_sim;
    - queries Genius API for lyrics to specified song. If found, lyrics are tokenized and put into Counter;
    otherwise, None is returned
    """
    while True: #queries sometimes throws random errors
        try:
            s = genius.search_song(query_name, query_artist, get_full_info = False)
            break
        except:
            pass
    if s: #lyrics found
        if match(query_artist, s.artist): #check to see if correct song retrieved
            song_lyrics = s.to_text().lower()
            song_lyrics = re.sub(r'[\(\[].*?[\)\]]', '', song_lyrics) #remove identifiers like chorus, verse, etc
            tokens = [t for t in tokenizer.tokenize(song_lyrics) if t not in punct] #don't want puncutation
            cnt = Counter(tokens)
            return cnt
        else: #wrong song retrieved
            print(f"{query_artist}: {query_name} not found on Genius.")            
    else: #lyrics not found
        print(f"{query_artist}: {query_name} not found on Genius.")


def lyrics_sim(query_lyrics_cnt, inv_idx, idf_dict, song_norms_dict):
    """
    @params: 
        query_lyrics_cnt: Counter of queried song's tokenized lyrics
        inv_idx: dict, {token:[(uri1, # of occurrences of token in song1), ...]}
        idf_dict: dict, {token:inverse document frequency value}
        song_norms_dict: dict, {uri:norm}
    @returns:
        dict of cosine similarity scores
    
    - Fast cosine implementation
    """
    
    query_tf_dict = query_lyrics_cnt 

    query_tfidf = dict()
    query_norm = 0
    for t in query_tf_dict: #creates tfidf dict for queried song's lyrics and computes its norm
        if t in idf_dict:
            tfidf = query_tf_dict[t] * idf_dict[t]
            query_tfidf[t] = tfidf
            query_norm += tfidf**2
    query_norm = np.sqrt(query_norm)
    
    doc_scores = dict() # uri :  cosine similarity
    for t in query_tfidf:
        for doc_id, tf in inv_idx[t]:
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (tf*idf_dict[t] * query_tfidf[t]) #doc_tfidf * query_tfidf
    for doc_id in doc_scores:
        doc_scores[doc_id] /= (song_norms_dict[doc_id] * query_norm) #normalize by doc_norm * query_norm
    
    return doc_scores

def get_song_uri(query_artist, query_name, sp):
    """
    @params: 
        query_artist: String
        query_name: String
        sp: SpotifyClient object
    @returns:
        String (uri of song) or None
    
    - helper function for af_sim;
    - queries Spotify API for URI of specified song. If not found, returns None
    """
    search_results = sp.search(f"{query_artist} {query_name}")['tracks']['items']
    if not search_results:
        print("Song not found on Spotify.")
    else:
        for d in search_results: #check each match
            artists = ",".join([x['name'] for x in d['artists']])
            if match(query_artist, artists) and match(query_name, d['name']): #match found
                return d['uri']
    print("Song not found on Spotify")

def get_audio_features(uri, sp):
    """
    @params: 
        uri: String
        sp: SpotifyClient object
    @returns:
        dict of song's audio features or None
    
    - helper function for af_sim;
    - queries Spotify API for audio features of specified song. If not found, returns None
    """
    data = sp.audio_features(uri)[0]
    if not data:
        return
    af = {k:data[k] for k in AF_COLS} #only get relevant fields

    track_info = sp.track(uri) #get artist and name of song
    af['artist'] = ", ".join([x['name'] for x in track_info['artists']])
    af['name'] = track_info['name']
    af['uri'] = data['uri']
    return af

def af_sim(query_af, af_matrix, af_song_norms, ix_to_uri, scaler, indices = None):
    """
    @params: 
        query_af: dict of queried song's audio features
        af_matrix: Numpy array of audio features (n_songs x n_audio_features)
        af_song_norms: Numpy array of audio feature norms (1 x n_songs)
        ix_to_uri: dict of integer index to song URI
        scaler: fitted StandardScaler object
    @returns:
        dict of cosine similarity scores
    
    - vectorized cosine similarity function
    """
    query_vec = scaler.transform(np.array([query_af[x] for x in AF_COLS]).reshape(1, -1)) #normalize features
    query_norm = np.linalg.norm(query_vec)
    

    scores = af_matrix.dot(query_vec.squeeze())/(query_norm * af_song_norms) #vectorized cosine similarity computation
    if indices: # only computing for a subset of the dataset
        scores_dict = {ix_to_uri[i]:scores[i] for i in indices}
    else: 
        scores_dict = {ix_to_uri[i]:scores[i] for i in range(len(scores))} 
    
    return scores_dict #dict of uri : cosine sim


def main(query, lyrics_weight, n_results, is_uri = False):
    """
    @params: 
        query: String; either a song's URI or its artist and name (should be in the form of "artist | name")
        lyrics_weight: float, between [0.0, 1.0] representing weight given to lyrics when computing similarity
        n_results: int, number of results to be returned
        is_uri: Boolean, True if inputted query is a URI, False otherwise

    @returns:
        dict of queried song's audio features
        List of tuples [(ith song's averaged similarity score, ith song's audio features) ...]
        List of ints [ith song's lyric similarity, ...]
    
    - main function user will interact with
    """
    start = time.time()
    sp = Spotify_Client() #re-initialize both API clients each time function is run to avoid timeout errors
    genius = Genius('bVEbboB9VeToZE48RaiJwrnAGLz8VbrIdlqnVU70pzJXs_T4Yg6pdPpJrTQDK46p')
    genius.verbose = False

    #TODO: update lyrics sim so that input is queried song's lyrics instead of artist and name 

    if not is_uri: #query is artist and name 
        query_artist, query_name = [x.split().lower() for x in query.split("|")]
        query_name = strip_name(query_name).lower()
        query_uri = get_song_uri(query_artist, query_name, sp)
        if not query_uri: #song not found on Spotify
            print("Song not found on Spotify")
            return
    else: #query is uri
        query_uri = query

    query_af = get_audio_features(query_uri, sp) #get queried song's audio features
    if not query_af: #audio features missing
        print("Song audio features not found on Spotify")
        return

    if is_uri: #if uri passed in, then get song's artist and name
        query_artist = query_af['artist'].split(",")[0].strip().lower()
        query_name = strip_name(query_af['name']).lower()

    if lyrics_weight == 0:
        sorted_lyric_sims = np.zeros(n_results)
        af_sim_scores = af_sim(query_af, vars_dict['af_matrix'], vars_dict['af_song_norms'], vars_dict['ix_to_uri'], vars_dict['scaler'])
    else:
        lyric_sim_scores = lyrics_sim(query_artist, query_name, genius, vars_dict['inv_idx'], vars_dict['idf_dict'], vars_dict['song_norms_dict'])
        uri_subset = lyric_sim_scores.keys()
        subset_indices = [vars_dict['uri_to_ix'][uri] for uri in uri_subset]
        subset_af_matrix = vars_dict['af_matrix'][subset_indices, :]
        subset_af_song_norms = vars_dict['af_song_norms'][subset_indices]
        af_sim_scores = af_sim(query_af, subset_af_matrix, subset_af_song_norms, vars_dict['ix_to_uri'], vars_dict['scaler'], subset_indices)
    
    if lyrics_weight == 0:
        averaged_scores = af_sim_scores
    else:
        af_weight = 1- lyrics_weight
        averaged_scores = {k:(af_sim_scores[k] * af_weight) + (lyric_sim_scores[k] * lyrics_weight) for k in af_sim_scores}
    

    ranked = sorted(averaged_scores.items(), key = lambda x: (-x[1], x[0]))

    if ranked[0][0] == query_uri:
        ranked = ranked[1:n_results+1]
    else:
        ranked = ranked[:n_results]
    
    output = [(x[1], vars_dict['uri_to_song'][x[0]]) for x in ranked]
    if lyrics_weight != 0:
        sorted_lyric_sims = [lyric_sim_scores[d['uri']] for _,d in output]

    end = time.time()
    print(f"{n_results} results retrieved in {round(end-start, 2)} seconds")
    return query_af, output, sorted_lyric_sims

