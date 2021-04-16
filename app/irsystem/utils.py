import unidecode #pylint: disable=import-error


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
        if ix != -1:
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