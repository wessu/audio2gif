import httplib
import urllib
import requests
import json
from PIL import Image
import os
# set the apikey and limit the # coming back
# apikey = "JZJ2NI81QT5O"  # test value
apikey = 'LIVDSRZULELA'
lmt = 50


# load the user's anonymous ID from cookies or some other disk storage
# anon_id = <from db/cookies>

# ELSE - first time user, grab and store their the anonymous ID


#if r.status_code == 200:
#    anon_id = json.loads(r.content)["anon_id"]
    # store in db/cookies for re-use later
#else:
#    anon_id = ""

# partial search

def extractFrames(inGIF):
	frames = Image.open(inGIF)
	basename = os.path.splitext(inGIF)[0]
	maxframes = 15
	i = 0
	os.mkdir(basename)
	crop_size = 240
        while frames and i <= maxframes:
		# crop to 240 * 240
		
                half_the_width = frames.size[0] / 2
		half_the_height = frames.size[1] / 2
		img = frames.resize((crop_size, crop_size))

		img.save(os.path.join(basename,'{}.jpg'.format(i)), 'GIF')
		i+= 1
		try:
			frames.seek(i)
		except EOFError:
			break

search = "dog"
lmt = 50

url = "https://api.tenor.com/v1/search?q={}&key=LIVDSRZULELA&limit={}&anon_id=3a76e56901d740da9e59ffb22b988242".format(search, lmt)

r = requests.get(url)
min_dim = 240

if r.status_code == 200:
    # return the search suggestions
    search_suggestion_list = json.loads(r.content)["results"]
    #print search_suggestion_list
    count = 0
    for result in search_suggestion_list:
    	result = result['media'][0]
    	if 'nanogif' in result:
    		if result['mediumgif']['dims'][0] >= min_dim and result['mediumgif']['dims'][1] >= min_dim:
    			print "saving image"
    			gif_url = result['mediumgif']['url']
    			urllib.urlretrieve(gif_url, 'gifs/{}_{}.gif'.format(search, count))
    			extractFrames('gifs/{}_{}.gif'.format(search, count))
    			count += 1
else:
    # handle a possible error
    search_suggestion_list = []

