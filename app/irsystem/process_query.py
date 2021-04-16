class QueryProcessor:
    def __init__(self, context):
        self.context = context

    """
    Takes in the name of a song and returns the top 10 ranked results. 
        query: String name of a song
        Raises an error if the queried song does not exist in the IR
    """
    def process_query(self, query):
        raise NotImplementedError()
