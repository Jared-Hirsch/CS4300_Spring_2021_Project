import pickle
from app.irsystem.SimSongs import SimilarSongs

class QueryProcessor:
    def __init__(self, stopwords_path, vars_dict_path, sp_path=None, gn_path=None, sp_username=None, sp_client_id=None, sp_client_secret=None, gn_token=None):
        self.vars_dict = pickle.load(open(vars_dict_path, 'rb'))
        self.stopwords = pickle.load(open(stopwords_path, 'rb'))
        self.spotify_path = sp_path
        self.genius_path = gn_path
        self.sim = SimilarSongs(self.stopwords, self.vars_dict, sp_path, gn_path, sp_username, sp_client_id, sp_client_secret, gn_token)
    

    def process_query(self, query, lyrics_weight, features_weights, n_results, requery_params, is_uri):
        """
        @returns:
            dict of queried song's audio features
            List of tuples [(ith song's averaged similarity score, ith song's audio features) ...]
            List of ints [ith song's lyric similarity, ...]
        """
        return self.sim.main(query, lyrics_weight, features_weights, n_results, requery_params, is_uri)
