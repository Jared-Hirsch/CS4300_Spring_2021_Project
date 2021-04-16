from . import *
import app.irsystem.constants as constants
from app.irsystem.is_context import ISContext
from app.irsystem.process_query import QueryProcessor

# ISContext
path_to_ir = ''
path_to_songs = ''
is_context = ISContext(path_to_ir, path_to_songs)
processor = QueryProcessor(is_context)


@irsystem.route('/', methods=['GET'])
def index():
    return render_template(constants.INDEX, name=constants.PROJECT_NAME, netids=constants.NETIDS, available_songs=is_context.songs['name'])


@irsystem.route('/results', methods=['GET'])
def search():
    query = request.args.get(constants.INPUT_QUERY)

    # Calculate results from the query
    results = processor.process_query(query)

    return render_template(constants.RESULTS, name=constants.PROJECT_NAME, netids=constants.NETIDS, query=query, available_songs=is_context.songs['name'], results=results)
