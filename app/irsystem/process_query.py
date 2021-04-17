import pickle
import app.irsystem.sim as sim


class QueryProcessor:
    def __init__(self, vars_dict_path):
        self.vars_dict = pickle.load(open(vars_dict_path, 'rb'))

    """
    Takes in the name of a song and returns the top 10 ranked results. 
        query: String name of a song
        Raises an error if the queried song does not exist in the IR
    """
    def process_query(self, query, lyrics_weight, n_results, is_uri):
        """
        @returns:
            dict of queried song's audio features
            List of tuples [(ith song's averaged similarity score, ith song's audio features) ...]
            List of ints [ith song's lyric similarity, ...]
        """
        return sim.main(query, lyrics_weight, n_results, self.vars_dict, is_uri)
