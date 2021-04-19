import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
import unidecode # pylint: disable=import-error
import matplotlib.pyplot as plt # pylint: disable=import-error
from nltk.tokenize import TreebankWordTokenizer # pylint: disable=import-error
from lyricsgenius import Genius # pylint: disable=import-error
import re
import time
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import spotipy.util as util
from sp_client import Spotify_Client
import string
from app.irsystem.utils import *
# from utils import *
import os


punct = set(string.punctuation)
punct.update({"''", "``", ""})
tokenizer = TreebankWordTokenizer()
AF_COLS = ['acousticness', 'danceability',
       'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'time_signature', 'valence']

class SimilarSongs:
    def __init__(self, stopwords, vars_dict, sp_path=None, gn_path=None, sp_username=None, sp_client_id=None, sp_client_secret=None, gn_token=None):
        self.stopwords = stopwords
        self.vars_dict = vars_dict
        if sp_path is not None:
            self.stopwords = stopwords
            self.vars_dict = vars_dict
            self.sp_path = sp_path
            with open(gn_path, 'r') as f:
                self.gn_token = f.readline().strip()
        else:
            self.sp_path = None
            self.sp_username = sp_username
            self.sp_client_id = sp_client_id
            self.sp_client_secret = sp_client_secret
            self.gn_token = gn_token
    
    def extract_annotations(self, song_id, genius):
        ants = genius.song_annotations(song_id)
        if len(ants) == 0:
            return ""
        
        out = []
        for line in ants:
            for a in line[1]:
                out.append(a[0])
        return " ".join(out).lower()

    def retrieve_lyrics(self, query_artist, query_name, genius):
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
        artist_obj = genius.search_artist(query_artist, max_songs=0)
        if artist_obj is None:
            return
        else:
            song_name = strip_name(query_name)
            song_obj = artist_obj.song(song_name)
            if song_obj is None or not match(query_artist, song_obj.artist):
                return
            else:
                lyrics = song_obj.to_text().lower()
                lyrics += self.extract_annotations(song_obj.id, genius)
                lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
                lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
                tokens = [t for t in tokenizer.tokenize(lyrics) if t not in punct and t not in self.stopwords]
                cnt = Counter(tokens)
                return cnt

    def lyrics_sim(self, query_lyrics_cnt, inv_idx, idf_dict, song_norms_dict):
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

    def get_song_uri(self, query_artist, query_name, sp):
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
            return
        else:
            for d in search_results: #check each match
                artists = ",".join([x['name'] for x in d['artists']])
                if match(query_artist, artists) and match(query_name, d['name']): #match found
                    return d['uri']
        return

    def get_audio_features(self, uri, sp):
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
        af['artist_name'] = ", ".join([x['name'] for x in track_info['artists']])
        af['track_name'] = track_info['name']
        af['uri'] = data['uri']
        return af

    def af_sim(self, query_af, af_matrix, af_song_norms, ix_to_uri, scaler, indices = []):
        """
        @params: 
            query_af: dict of queried song's audio features
            af_matrix: Numpy array of audio features (n_songs x n_audio_features)
            af_song_norms: Numpy array of audio feature norms (1 x n_songs)
            ix_to_uri: dict of integer index to song URI
            scaler: fitted StandardScaler object
            indices: list of ints; indices of subset of songs that could be considered
        @returns:
            dict of cosine similarity scores
        
        - vectorized cosine similarity function
        """
        query_vec = scaler.transform(np.array([query_af[x] for x in AF_COLS]).reshape(1, -1)) #normalize features
        query_norm = np.linalg.norm(query_vec)
        

        scores = af_matrix.dot(query_vec.squeeze())/(query_norm * af_song_norms) #vectorized cosine similarity computation
        if indices: # only computing for a subset of the dataset
            #TODO: problem: scores has length = len(indices), but indices has values between [0, len(dataset)]
            #need to convert indices to score indices
            scores_dict = {ix_to_uri[i]:scores[i] for i in indices}
        else: 
            scores_dict = {ix_to_uri[i]:scores[i] for i in range(len(scores))} 
        
        return scores_dict #dict of uri : cosine sim


    def main(self, query, lyrics_weight, n_results, is_uri = False):
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

        #re-initialize both API clients each time function is run to avoid timeout errors
        if self.sp_path is None:
            sp = Spotify_Client(self.sp_username, self.sp_client_id, self.sp_client_secret)
        else:
            sp = Spotify_Client(self.sp_path)
        genius = Genius(self.gn_token, verbose = False, retries = 5)




        if not is_uri: #query is artist and name 
            query_artist, query_name = [x.strip().lower() for x in query.split("|")]
            query_name = strip_name(query_name)
            query_uri = self.get_song_uri(query_artist, query_name, sp)
            if not query_uri: #song not found on Spotify
                raise ValueError("Invalid search: " + query)
        else: #query is uri
            query_uri = query

        query_af = self.get_audio_features(query_uri, sp) #get queried song's audio features
        if not query_af: #audio features missing
            raise ValueError("Song audio features not found on Spotify for " + query)

        if is_uri: #if uri passed in, then get song's artist and name
            query_artist = query_af['artist_name'].split(",")[0].strip().lower()
            query_name = strip_name(query_af['track_name']).lower()

        if lyrics_weight == 0: #don't consider lyrics at all; compute audio feature similarity across all songs in dataset
            sorted_lyric_sims = np.zeros(n_results)
        else:
            query_lyrics_cnt = self.retrieve_lyrics(query_artist, query_name, genius)
            if not query_lyrics_cnt:
                raise ValueError("Song lyrics not found on Genius for " + query)

            lyric_sim_scores = self.lyrics_sim(query_lyrics_cnt, self.vars_dict['inv_idx'], self.vars_dict['idf_dict'], self.vars_dict['song_norms_dict'])
        af_sim_scores = self.af_sim(query_af, self.vars_dict['af_matrix'], self.vars_dict['af_song_norms'], self.vars_dict['ix_to_uri'], self.vars_dict['scaler']) 

            
            
            #TODO: only compute audio feature similarity on songs with nonzero lyrical similarity

            # uri_subset = lyric_sim_scores.keys() #if consider lyrics, then only compute audio feature similarity for songs with nonzero lyric similarity
            # subset_indices = [vars_dict['uri_to_ix'][uri] for uri in uri_subset]
            # subset_af_matrix = vars_dict['af_matrix'][subset_indices, :]
            # subset_af_song_norms = vars_dict['af_song_norms'][subset_indices]
            # af_sim_scores = af_sim(query_af, subset_af_matrix, subset_af_song_norms, vars_dict['ix_to_uri'], vars_dict['scaler'], subset_indices)
        
        if lyrics_weight == 0: 
            averaged_scores = af_sim_scores
        else: #if considering lyrics, then take weighted average of audio feature and lyrical similarity scores
            af_weight = 1 - lyrics_weight
            averaged_scores = {k:(af_sim_scores[k] * af_weight) + (lyric_sim_scores[k] * lyrics_weight) for k in lyric_sim_scores}
        

        #TODO: handle different versions of same song in output (ex: "I'll Never Love Again - Film Version", "I'll Never Love Again - Extended Version")
        ranked = sorted(averaged_scores.items(), key = lambda x: (-x[1], x[0])) #sort songs in descending order of similarity scores
        output = []
        cnt = 0
        i = 0
        while i < len(ranked) and cnt < n_results:
            uri, score = ranked[i][0], ranked[i][1]
            song_data = self.vars_dict['uri_to_song'][uri]
            if uri != query_uri and not match(song_data['artist_name'], query_artist) and not match(song_data['track_name'], query_name): 
                #don't want to return inputted/different versions of inputted song
                output.append((score, song_data))
                cnt += 1
            i += 1


        if lyrics_weight != 0: #if considering lyrics, then sort lyrical similarity scores in same order as output
            sorted_lyric_sims = [lyric_sim_scores[d['track_id']] for _,d in output] #TODO: change track_id to uri

        end = time.time()
        print(f"{n_results} results retrieved in {round(end-start, 2)} seconds")
        return query_af, output, sorted_lyric_sims

def print_results(output, indent = True):
    out = []
    for score, data in output:
        song_info = f"{data['artist_name']} | {data['track_name']}"
        out.append(f"({round(score, 4)}) {song_info}")
    if indent:
        print("\t" + "\n\t".join(out))
    else:
        print("\n".join(out))

if __name__ == "__main__":
    path = os.getcwd() + os.path.sep
    stopwords = pickle.load(open(path + 'stopwords.pkl', 'rb'))
    sp_path = path + 'token.txt'
    gn_path = path + 'genius_token.txt'

    vars_path = os.getcwd() + os.path.sep + '..' + os.path.sep + '..' + os.path.sep + 'sample_data' + os.path.sep
    vars_dict = pickle.load(open(vars_path + 'top_annotations_sim_vars.pkl', 'rb'))

    SimSongs = SimilarSongs(stopwords, vars_dict, sp_path, gn_path)

    query = 'The Chainsmokers | Closer'
    lyrics_weight = 0
    n_results = 10
    is_uri = False
    query_af, output, _ = SimSongs.main(query, lyrics_weight, n_results, is_uri)
    print(f"Results for: {query_af['artist_name']} | {query_af['track_name']}")
    print_results(output)

#     query = 'spotify:track:0rKtyWc8bvkriBthvHKY8d'
#     lyrics_weight = 0
#     n_results = 10
#     is_uri = True
#     query_af, output, _ = main(query, lyrics_weight, n_results, vars_dict, is_uri)
#     print(f"Results for: {query_af['artist_name']} | {query_af['track_name']}")
#     print_results(output)
