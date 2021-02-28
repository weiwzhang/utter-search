import pandas as pd
import typing
import functools

# process taskmaster datasets
def process_tm1(
		filename: str, 
		clean_label_func: typing.Callable[[pd.DataFrame], pd.DataFrame], 
		get_utter_func: typing.Callable[[dict], str], 
		blocklist: [str]) -> pd.DataFrame:
	df = pd.read_json(filename)[['instruction_id', 'utterances', 'conversation_id']]
	# process utterance
	df['utterance'] = df['utterances'].apply(get_utter_func)
	df.drop('utterances', axis=1, inplace=True)
	df = df.dropna(how='any', axis = 0)
	print("df shape after dropping null values:", df.shape)
	# process label
	df['label'] = df['instruction_id'].apply(clean_label_tm1)
	print("df shape:", df.shape)
	print(df.head(5))
	df = df.dropna(how='any', axis = 0)
	print("df shape after dropping null label values:", df.shape)
	df = df[~df['conversation_id'].isin(blocklist)]
	print("df shape after removing blocklist:", df.shape)
	df.drop_duplicates(subset=['utterance'], inplace=True)
	print("df shape after removing duplicated utterancse:", df.shape)
	# df.to_csv('test.csv')
	return df

def process_tm2(
		filename: str, 
		label: str, 
		get_utter_func: typing.Callable[[dict], str], 
		blocklist: [str]) -> pd.DataFrame:
	df = pd.read_json(filename)[['instruction_id', 'utterances', 'conversation_id']]
	# process utterance
	df['utterance'] = df['utterances'].apply(get_utter_func)
	df.drop('utterances', axis=1, inplace=True)
	df = df.dropna(how='any', axis = 0)
	print("df shape after dropping null values:", df.shape)
	# process label
	df['label'] = label
	print("df shape:", df.shape)
	print(df.head(5))
	df = df.dropna(how='any', axis = 0)
	print("df shape after dropping null label values:", df.shape)
	df = df[~df['conversation_id'].isin(blocklist)]
	print("df shape after removing blocklist:", df.shape)
	df.drop_duplicates(subset=['utterance'], inplace=True)
	print("df shape after removing duplicated utterancse:", df.shape)
	# df.to_csv('test.csv')
	return df

# utils
def clean_label_tm1(i):
	if i.startswith("movie-ticket"):
		return "movie-ticket"
	if i.startswith("movie-find"):
		return "movie-search"
	if i.startswith("restaurant-table-1"):
		return "restaurant-search"
	if i.startswith("restaurant-table"):
		return "restaurant-book"
	if i.startswith("pizza"):
		return "pizza-order"
	if i.startswith("coffee"):
		return "coffee-order"
	if i.startswith("auto-repair"):
		return "auto-repair-appt"
	if i.startswith("uber"):
		return "ride-book"

def get_tm1woz_user_utterance(u):
	for i in u:
		# if i['speaker'] == 'USER':
		# 	print(i['index'])
		# 	return i['text']
		if 'segments' in i and "annotations" in i['segments'][0]:
			# print(i['index'])
			return i['text']

def get_tm2_user_utterance(u):
	first_user, first_user_annotated = "", "" 
	for i in u:
		# get first user utterance
		if i['speaker'] == 'USER' and not first_user:
			first_user = i['text']
		# first user speech that has annotation
		if i['speaker'] == 'USER' and 'segments' in i and not first_user_annotated: 
			first_user_annotated = i['text']
		# return longer utterance of the two
		if first_user and first_user_annotated:
			return first_user if len(first_user) > len(first_user_annotated) else first_user_annotated

master_labels = ["flight-search", "food-order", "hotel-search", "music-search", "sports-stats", "restaurant-search", "movie-ticket", "pizza-order", "coffee-order", "auto-repair-appt", "ride-book", "movie-search", "restaurant-book"]
files_tm1 = [
	'./data/tm1-written.json', 
	# './data/tm1-woz.json', # very problematic
]
files_tm2 = [
	'./data/tm2-flights.json', 
	'./data/tm2-food_ordering.json',
	# './data/tm2-hotels.json', # problematic
	'./data/tm2-movies.json',
	'./data/tm2-music.json',
	'./data/tm2-restaurant_search.json',
	# './data/tm2-sports.json' #separate sub-categories
]
files_tm3 = []
get_utter_funcs_tm1 = [
	lambda x: x[0]['text'],
	# get_tm1woz_user_utterance,
]
get_utter_funcs_tm2 = get_tm2_user_utterance
get_utter_funcs_tm3 = []
blocklists_tm1 = [] 
blocklists_tm2 = [
	[],
	# [],
	[],
	[],
	# [],
	["dlg-05ed0ef0-8851-4467-a44c-64ad10909c40", "dlg-081ad4bc-5f15-490e-b0f5-6034cb3c5e64", "dlg-d90bf99e-a0fc-4af7-9de2-23675b4926e3", "dlg-68f6d0a5-ca8c-4a78-bb3a-fe0983e4609f", "dlg-94503c6f-6a6a-4423-b541-440468f826f0", "dlg-99740651-8f95-4894-bf06-3c4f6d7c936e", "dlg-9a6f224a-03df-4411-8ce1-36dcf0ceee7d", "dlg-9cee8c61-f377-435f-9fbe-060487eba83f", "dlg-9d3bb23e-1427-4fb9-a804-43e9d08117a2", "dlg-ac143827-f1d8-4e66-bf37-f5009620d1d7", "dlg-cce7982c-a8cb-4226-8917-5d95197cc6d4", "dlg-ff90ae1c-1f25-44f2-b67e-30880ecd795b"],
	["dlg-794cf26f-47a9-45b9-8a9d-9acf73dbc4c0"],
	["dlg-00bb90c6-0968-4e40-880b-0857f1692b0b", "dlg-1f98aff5-5f40-45d7-89c9-92337b86ee3b", "dlg-1ffe0197-1b6b-42c1-b268-dabb0ba3c89e", "dlg-2272a087-9ad7-41a2-a4f8-2812cd9d3ee4", "dlg-35cd469f-810c-4a78-b83c-026a318edb2d", "dlg-38a6bb20-656d-4a88-8c6d-4c7414a73db0", "dlg-3fd7f477-749d-4da9-9482-3c52581540ab", "dlg-520ce320-76b5-4d90-9ad3-36c54bb1aa41", "dlg-6477075c-8b40-4a0d-8d8c-2586a8ab0988", "dlg-6b28efd8-7fad-41f8-b158-ef55af43ed71", "dlg-6fa6c055-e018-4549-8032-efb2aed8966e", "dlg-763bce81-2d84-481e-aa70-4ea177db2dc7", "dlg-76146338-d73a-4cd9-8f6e-77074b189217", "dlg-7a255873-0bf2-4557-9c5b-f4086ebb79f6", "dlg-83d02774-4061-481a-8a8a-f486772d67de", "dlg-83e6d390-83d3-4671-bb2c-c73b6c5ca992", "dlg-8a18e51a-6731-4e98-ba87-f1b2d4f1fce1", "dlg-8af52ea6-39b8-491d-8bfb-10a936aa4e57", "dlg-8e5c680a-96b1-4ea1-8259-4e93937f7f3f", "dlg-a4bdd453-629d-4369-a5de-00504b5fd0ed", "dlg-aa2942a1-fe31-43e9-8d72-86bd20ba8499", "dlg-b75e2007-a9fd-4c45-9f5b-e492951b14d4", "dlg-c4e06448-323a-4ff6-b18c-09c0c9a5bffe", "dlg-cb2b08ce-33d4-4c0e-aab4-cd6e38a682d2", "dlg-cca3b925-acce-48af-85c8-b8ef3b4dd8a8", "dlg-d078d498-9dc2-4031-8207-d254a57232b4", "dlg-d6052d56-dc09-4d84-b2b1-9f8b995cba72", "dlg-db5b8a06-c062-4a4f-8135-2f26b5e2d168", "dlg-dc8f8215-6755-4b39-95f1-9727f9d39320", "dlg-dce83840-2486-459f-b087-33ffb1a4d54e", "dlg-e07e2516-07c3-4f70-bc47-772e8aea6a0f", "dlg-e13f35df-1136-462d-ac73-048635e414ff"]
	# ["dlg-552101bd-5de8-4347-b12e-7323b1dcbd00", "dlg-805e5bbd-9054-4a13-9313-dd011fbe7d2f", "dlg-812f86e8-32b8-4348-967a-bf5264002813", "dlg-d7946a1c-e3af-4e9c-b02d-00ad2cae3007", "dlg-f193ff0d-9433-48f0-a1d9-6003b318c7fe"]
]
blocklists_tm3 = []
labels_tm2 = [
	"flight-search", 
	"food-order", 
	# "hotel-search", 
	"movie-search", 
	"music-search", 
	"restaurant-search",
	# "sports-stats"
]

def main():
	dfs = []
	for i in range(len(files_tm1)):
		df = process_tm1(files_tm1[i], clean_label_tm1, get_utter_funcs_tm1[i], blocklists_tm1)
		dfs.append(df)

	for i in range(len(files_tm2)):
		df = process_tm2(files_tm2[i], labels_tm2[i], get_utter_funcs_tm2, blocklists_tm2[i])
		dfs.append(df)
	
	print("len(dfs): ", len(dfs))
	final_data = pd.concat(dfs)
	print("finally, data size is:", final_data.shape)
	final_data.to_csv('data.csv')

main()
