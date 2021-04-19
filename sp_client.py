import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials


def Spotify_Client(sp_path):
    with open(sp_path, "r") as f:
        username, client_id, client_secret = [x.strip() for x in f]
    redirect_uri = "http://localhost:8000"
    scope = "playlist-read-private playlist-read-collaborative user-read-recently-played playlist-modify-private user-library-read user-top-read user-library-modify user-modify-playback-state streaming"
    token = util.prompt_for_user_token(username,scope,client_id,client_secret,redirect_uri)
    sp = spotipy.Spotify(auth=token)
    return sp

def Spotify_Client(username, client_id, client_secret):
    redirect_uri = "http://localhost:8000"
    scope = "playlist-read-private playlist-read-collaborative user-read-recently-played playlist-modify-private user-library-read user-top-read user-library-modify user-modify-playback-state streaming"
    auth_manager = SpotifyClientCredentials(client_id, client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

