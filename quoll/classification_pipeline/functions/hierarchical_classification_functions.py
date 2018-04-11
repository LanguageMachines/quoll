
from collections import defaultdict

def setup_second_layer(trainlabels1, predictions1):
	# All specific labels will form the basis of the second classification layer 
	# e.g.: the second layer labels linked to a particular first layer label will be 
	# modeled separately in the second layer
	label_exps = defaultdict(lambda : defaultdict(list)): # dictionary to keep track of layer1 labels
	# set up second layer trainlabels to train on
	for i,label in enumerate(trainlabels1):
		label_exps[label]['trainindices'].append(str(i))
	# set up second layer test labels to train on (based on predictions on first labels --> errors will spill over)
	for i,label in enumerate(predictions1):
		label_exps[label]['testindices'].append(str(i))
	return label_exps
