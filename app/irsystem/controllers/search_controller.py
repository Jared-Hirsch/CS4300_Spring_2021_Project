from . import *
import os
import app.irsystem.constants as constants
from app.irsystem.process_query import QueryProcessor
from app.irsystem.sim_preprocess import set_stopwords

vars_dict_path = os.getcwd() + os.path.sep + 'sample_data' + os.path.sep + 'top_annotations_sim_vars.pkl'
stopwords_path = os.getcwd() + os.path.sep + 'app' + os.path.sep + 'irsystem' + os.path.sep + 'stopwords.pkl'
set_stopwords(stopwords_path)
processor = QueryProcessor(stopwords_path, vars_dict_path)
songs = map(lambda s: s['track_name'], processor.vars_dict['uri_to_song'].values())


@irsystem.route('/', methods=['GET'])
def index(errors=[]):
    return render_template(constants.INDEX, name=constants.PROJECT_NAME, netids=constants.NETIDS, songs=songs, errors=errors)


@irsystem.route('/results', methods=['GET'])
def search():
    query = request.args.get(constants.INPUT_QUERY)
    print(query)
    if query == "" or query is None:
        return index(["Query cannot be empty empty"])

    # Calculate results from the query
    results = processor.process_query(query, 0.5, 0.5, False)

    return render_template(constants.RESULTS, name=constants.PROJECT_NAME, netids=constants.NETIDS, songs=songs, query=query, results=results)
