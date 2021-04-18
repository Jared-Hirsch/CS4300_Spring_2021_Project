import spotipy
import spotipy.util as util

def Spotify_Client(path):
    with open(path, "r") as f:
        username, client_id, client_secret = [x.strip() for x in f]
    redirect_uri = "http://localhost:8000"
    scope = "playlist-read-private playlist-read-collaborative user-read-recently-played user-library-read user-top-read"
    token = util.prompt_for_user_token(username,scope,client_id,client_secret,redirect_uri)
    sp = spotipy.Spotify(auth=token)
    return sp