import time

from . import *
import os
import spotipy
import app.irsystem.constants as constants
from app.irsystem.process_query import QueryProcessor
from config import Config

vars_dict_path = os.getcwd() + os.path.sep + 'sample_data' + \
    os.path.sep + '12000_sim_vars.pkl'
stopwords_path = os.getcwd() + os.path.sep + 'app' + os.path.sep + \
    'irsystem' + os.path.sep + 'stopwords.pkl'
processor = QueryProcessor(stopwords_path, vars_dict_path, sp_username=Config.SP_USERNAME,
                           sp_client_id=Config.SP_CLIENT_ID, sp_client_secret=Config.SP_CLIENT_SECRET, gn_token=Config.GENUIS_TOKEN)
songs = map(lambda s: s['track_name'],
            processor.vars_dict['uri_to_song'].values())


@irsystem.route('/', methods=['GET'])
def index(errors=[]):
    logged_in = session.get('sp_token') is not None
    return render_template(constants.INDEX, name=constants.PROJECT_NAME, students=constants.NAMES, audiofeatures=constants.AUDIO_FEATURES, songs=songs, errors=errors, logged_in = logged_in)


@irsystem.route('/results', methods=['GET'])
def search():
    # Get required inputs
    query = request.args.get(constants.INPUT_QUERY)
    lyr_sim = request.args.get(constants.LYRICAL_SIMILARITY) 
    num_songs = int(request.args.get(constants.NUM_SONGS))
    logged_in = session.get('sp_token') is not None

    # Get weights for each audio feature
    features_weights = []
    for af in constants.AUDIO_FEATURES:
        arg = request.args.get(af[1])
        if arg is None:
            return index([af[0]+" is missing"])
        features_weights.append(int(request.args.get(af[1]))/100)

    # Get requery parameters if available
    requery_params = {}
    for af in constants.AUDIO_FEATURES:
        arg = request.args.get(af[1] + 'rq', '')
        if arg == '':
            requery_params = None
            break
        requery_params[af[0].lower()] = (float(arg))
    if request.args.get(constants.LIKED):
        liked = [l for l in request.args.get(constants.LIKED).split(',') if l]
    else:
        liked = []
    if request.args.get(constants.DISLIKED):
        disliked = [d for d in request.args.get(constants.DISLIKED).split(',') if d]
    else:
        disliked = []

    # Check for Errors in input
    if query == "" or query is None:
        return index(["Select a song to search on"])
    if lyr_sim is None:
        return index(["Must include audio and lyrical similarity value"])
    if num_songs is None or num_songs > 20 or num_songs < 1:
        return index(["Choose a number of songs to be returned between 1 and 20"])

    # Calculate results from the query
    try:
        print('liked' + str(liked))
        print('disliked' + str(disliked))
        print('query' + str(query))
        print('requery_params' + str(requery_params))
        query_af, output, lyr, af_scores = processor.process_query(
            query, int(lyr_sim)/100, features_weights, num_songs, requery_params, liked, disliked, False)
        output = [(str(round(sim, 3)).ljust(5, '0'), af)
                  for (sim, af) in output]
        lyr = [str(round(sim, 3)).ljust(5, '0') for sim in lyr]
        af_scores = [str(round(sim, 3)).ljust(5, '0') for sim in af_scores]

        results = query_af, output, lyr, af_scores
    except ValueError as err:
        print(str(err))
        return index([str(err)])

    # Dummy results value
    # results = {'track_name': 'Celebration', 'artist_name': 'Kanye West'}, \
    #     [(10, {'track_name': 'Celebration', 'artist_name': 'Kanye West'}),
    #      (20, {'track_name': 'Late', 'artist_name': 'Kanye West'}),
    #      ((30, {'track_name': 'Addiction', 'artist_name': 'Kanye West'}))], \
    #     [1, 2, 3]

    return render_template(
        constants.RESULTS, name=constants.PROJECT_NAME, students=constants.NAMES, audiofeatures=constants.AUDIO_FEATURES,
         songs=songs, query=query, results=results, liked=liked, disliked=disliked, logged_in=logged_in)

@irsystem.route("/auth_sp", methods=['GET'])
def authorize_spotify():
    sp_oauth = spotipy.oauth2.SpotifyOAuth(client_id=Config.SP_CLIENT_ID,
                                           client_secret=Config.SP_CLIENT_SECRET,
                                           redirect_uri=Config.REDIRECT_URI,
                                           scope=constants.SPOTIFY_SCOPE)
    sp_auth_url = sp_oauth.get_authorize_url()

    return redirect(sp_auth_url)

@irsystem.route("/callback", methods=['GET', 'POST'])
def spotify_callback():
    sp_oauth = spotipy.oauth2.SpotifyOAuth(client_id=Config.SP_CLIENT_ID,
                                           client_secret=Config.SP_CLIENT_SECRET,
                                           redirect_uri=Config.REDIRECT_URI,
                                           scope=constants.SPOTIFY_SCOPE)

    session.clear()

    token_info = sp_oauth.get_access_token(request.args.get('code'))
    session['sp_token'] = token_info

    return redirect('/')

@irsystem.route('/create_playlist', methods=['POST'])
def create_playlist():
    token_info = validate_token(session)
    if token_info is None:
        return index(["Spotify Not logged in"])

    sp = spotipy.Spotify(auth=token_info.get('access_token'))

    if not request.json:
        return {'Error': 'JSON not provided'}, 422, {'Conent-Type': 'application/json'}
    jdata = request.get_json()
    print(jdata)
    playlist_name = jdata['name']
    is_private = jdata['isprivate']
    songs = jdata['songs']

    return "", 200

@irsystem.route('/logout', methods=['GET'])
def logout():
    session.clear()

    return redirect('/')


# Validates and returns spotify authentication token or refreshes an existing one
def validate_token(session):
    token_info = session.get('sp_token', {})

    if token_info is None:
        return None

    now = int(time.time())
    is_token_expired = token_info['expires_at'] - now < 10

    if is_token_expired:
        sp_oauth = spotipy.oauth2.SpotifyOAuth(client_id=Config.SP_CLIENT_ID,
                                               client_secret=Config.SP_CLIENT_SECRET,
                                               redirect_uri=Config.REDIRECT_URI,
                                               scope=constants.SPOTIFY_SCOPE)
        token_info = sp_oauth.refresh_access_token(session['sp_token']['refresh_token'])

    return token_info
