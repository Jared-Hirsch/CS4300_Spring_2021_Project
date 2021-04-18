import pickle
from app.irsystem.SimSongs import SimilarSongs

class QueryProcessor:
    def __init__(self, stopwords_path, vars_dict_path, spotify_path, genius_path):
        self.vars_dict = pickle.load(open(vars_dict_path, 'rb'))
        self.stopwords = pickle.load(open(stopwords_path, 'rb'))
        self.spotify_path = spotify_path
        self.genius_path = genius_path
        self.sim = SimilarSongs(self.stopwords, self.vars_dict, spotify_path, genius_path)
    

    def process_query(self, query, lyrics_weight, n_results, is_uri):
        """
        @returns:
            dict of queried song's audio features
            List of tuples [(ith song's averaged similarity score, ith song's audio features) ...]
            List of ints [ith song's lyric similarity, ...]
        """
        return self.sim.main(query, lyrics_weight, n_results, is_uri)
