from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Twitter Account Recommendations"
net_ids = ["Jared Hirsch: jsh328", "Ronnie Dumesh: rad338", "Ravina Patel: rpp62", "Chris Chen: cmc447"]

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		data = range(5)
	return render_template('search.html', name=project_name, netids=net_ids, output_message=output_message, data=data)
