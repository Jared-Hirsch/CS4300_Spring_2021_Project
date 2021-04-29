from . import *
import os
import app.irsystem.constants as constants
from app.irsystem.process_query import QueryProcessor
from config import Config
# import math

vars_dict_path = os.getcwd() + os.path.sep + 'sample_data' + \
    os.path.sep + 'top_annotations_sim_vars.pkl'
stopwords_path = os.getcwd() + os.path.sep + 'app' + os.path.sep + \
    'irsystem' + os.path.sep + 'stopwords.pkl'
processor = QueryProcessor(stopwords_path, vars_dict_path, sp_username=Config.SP_USERNAME,
                           sp_client_id=Config.SP_CLIENT_ID, sp_client_secret=Config.SP_CLIENT_SECRET, gn_token=Config.GENUIS_TOKEN)
songs = map(lambda s: s['track_name'],
            processor.vars_dict['uri_to_song'].values())


@irsystem.route('/', methods=['GET'])
def index(errors=[]):
    return render_template(constants.INDEX, name=constants.PROJECT_NAME, netids=constants.NETIDS, audiofeatures=constants.AUDIO_FEATURES, songs=songs, errors=errors)


@irsystem.route('/results', methods=['GET'])
def search():
    query = request.args.get(constants.INPUT_QUERY)
    lyr_sim = request.args.get(constants.LYRICAL_SIMILARITY)
    print(query)
    if query == "" or query is None or lyr_sim is None:
        return index(["Query cannot be empty empty, audio similarity cannot be missing, lyrical similarity cannot be missing"])

    # Calculate results from the query
    try:
        query_af, output, lyr = processor.process_query(
            query, int(lyr_sim)/100, 10, False)
        output = [(str(round(sim, 3)).ljust(5, '0'), af)
                  for (sim, af) in output[:(constants.NUM_RESULTS)]]
        results = query_af, output, lyr
    except ValueError as err:
        return index([str(err)])

    # Dummy results value
    # results = {'track_name': 'Celebration', 'artist_name': 'Kanye West'}, \
    #     [(10, {'track_name': 'Celebration', 'artist_name': 'Kanye West'}),
    #      (20, {'track_name': 'Late', 'artist_name': 'Kanye West'}),
    #      ((30, {'track_name': 'Addiction', 'artist_name': 'Kanye West'}))], \
    #     [1, 2, 3]

    return render_template(constants.RESULTS, name=constants.PROJECT_NAME, netids=constants.NETIDS, songs=songs, query=query, results=results)
