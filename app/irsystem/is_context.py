"""
The Context for the information system containing the IR and the Songs involved
"""
class ISContext:
    """
    Parses the context of the Information System by collecting the Intermediate Representation (Cosine Similarity Matrix)
    and the song related information.
        ir_location: The path to the csv file containing the Cosine Similarity Matrix
        songs_location: The path to the pickle file containing song information such as lyrics
    """
    def __init__(self, ir_location, songs_location):
        self.ir = self.parseCosineSimilarityMatrix(ir_location)
        self.songs = self.parseSongInformation(songs_location)

    def parseCosineSimilarityMatrix(self, path):
        raise NotImplementedError()

    def parseSongInformation(self, path):
        raise NotImplementedError()