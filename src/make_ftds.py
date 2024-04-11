import json
from tqdm import tqdm

with open('/home/cse/btech/cs1200448/MatLlama/finetune/custom-finetuning-data/val_ft.json', 'r') as f:
	data = json.load(f)

new_ds = []
for item in data:
	new_item = dict()
	if item['task']=='ner':
		if item['dataset']=='matscholar':
			new_item['system'] = 'You are a linguist and a material scientist. You need to identify the named entity for each of the keywords given after WORDS in the input. Answer to the question should be from one of the provided options. Do not output anything else other than the answer. You should output the word entity pair separated by ":" in each line. Your options are: b-mat, i-mat, b-spl, i-spl, b-dsc, i-dsc, b-pro, i-pro, b-apl, i-apl, b-smt, i-smt, b-cmt, i-cmt. Answer for each word must be in a new line.'
		if item['dataset']=='sofc_token':
			new_item['system'] = 'You are a linguist and a material scientist. You need to identify the named entity for each of the keywords given after WORDS in the input. Answer to the question should be from one of the provided options. Do not output anything else other than the answer. You should output the word entity pair separated by ":" in each line. Your options are: b-material, i-material, b-device, i-device, b-experiment, i-experiment, b-value, i-value. Answer for each word must be in a new line.'
		if item['dataset']=='sc_comics':
			new_item['system'] = 'You are a linguist and a material scientist. You need to identify the named entity for each of the keywords given after WORDS in the input. Answer to the question should be from one of the provided options. Do not output anything else other than the answer. You should output the word entity pair separated by ":" in each line. Your options are: material, doping, sc, value, process, characterization, element, property, main. Answer for each word must be in a new line.'
	if item['task']=='pc':
		new_item['system'] = "You are a linguist and a material scientist. You need to classify if the paragraph below is related to inorganic glass research. Answer to the question should be from one of the provided options. Do not output anything else other than the answer. Your choices are: yes/no."
	if item['task']=='sf':
		new_item['system'] = 'You are a linguist and a material scientist. You need to identify the slots of the keywords given after WORDS in the input. Answer to the question should be from one of the provided options. Do not output anything else other than the answer. You should output the word entity pair separated by ":" in each line. Your options are: i-device, b-voltage, b-anode_material, b-cathode_material, b-time_of_operation, i-working_temperature, b-conductivity, i-fuel_used, i-interlayer_material, i-time_of_operation, i-anode_material, i-current_density, b-degradation_rate, i-resistance, i-conductivity, b-current_density, b-working_temperature, i-thickness, i-experiment_evoking_word, b-open_circuit_voltage, i-degradation_rate, b-electrolyte_material, i-open_circuit_voltage, i-electrolyte_material, b-fuel_used, b-power_density, i-power_density, b-interlayer_material, b-thickness, b-device, b-experiment_evoking_word, i-cathode_material, b-resistance, i-support_material, i-voltage, b-support_material. Answer for each word must be in a new line.'
	if item['task']=='ee':
		new_item['system'] = "You are a linguist and a material scientist. Given the event type, the trigger word and the arguments of the event, identify the roles of the arguments for that event. The roles must be from the options given. Output each argument and role pair in a new line. Eg: <argument> : <role>\n<argument> : <role>. Your options for roles are: site, dopant. Do not output anything else."
	if item['task']=='re':
		if item['dataset']=='structured_re':
			new_item['system'] = "You are a linguist and a material scientist. You need to extract relation between the entities given in the text. Answer to the question should be from one of the provided options. Do not output anything else other than the answer. Your options are: coulombic efficiency, capacity, conductivity, voltage, energy."
		if item['dataset']=='sc_comics':
			new_item['system'] = "You are a linguist and a material scientist. You need to extract relation between the entities given in the text. Answer to the question should be from one of the provided options. Do not output anything else other than the answer. Your options are: target, condition, equivalent." 
	if item['task']=='sar':
		new_item['system'] = 'You are a linguist and a material scientist. You need to mark the synthesis action for each of the keywords given after WORDS in the input. Answer to the question should be from one of the provided options. Do not output anything else other than the answer. You should output the word entity pair separated by ":" in each line. Your options are: cooling, heating, mixing, non-altering, purification, reaction, shaping, starting. Answer for each word must be in a new line.'
	if item['task']=='sc':
		new_item['system'] = "You are a linguist and a material scientist. You need to classify if the paragraph below is related to solid oxide fuel cell (SOFC) research. Answer to the question should be from one of the provided options. Do not output anything else other than the answer. Your choices are: yes/no."
	new_item['question'] = item['input']
	new_item['answer'] = item['output']
	new_ds.append(new_item)

with open('/scratch/cse/btech/cs1200448/MatLlama/ft_ds/val_ft_new.json', 'w') as f:
	for document in tqdm(new_ds):  
		f.write(json.dumps(document) + "\n")

print(len(new_ds))
print(new_ds[0])	
