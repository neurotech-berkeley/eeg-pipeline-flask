from flask import Flask, request
import numpy as np
import pandas as pd
import category_encoders as ce
app = Flask(__name__)

@app.route('/encodeData', methods=['POST'])
def encodeData():
    infData = pd.read_csv('inf_data.csv', sep=',')

    infData.platform = infData.platform.fillna('X')
    infData.youtube_subs = infData.youtube_subs.fillna(0)
    infData.youtube_engagement = infData.youtube_engagement.fillna(0)
    infData.instagram_followers = infData.instagram_followers.fillna(0)
    infData.instagram_engagement = infData.instagram_engagement.fillna(0)
    infData.facebook_followers = infData.facebook_followers.fillna(0)
    infData.facebook_engagement = infData.facebook_engagement.fillna(0)
    infData.twitter_followers = infData.twitter_followers.fillna(0)
    infData.twitter_engagement = infData.twitter_engagement.fillna(0)
    infData.blog_traffic = infData.blog_traffic.fillna(0)
    infData.follower_country1 = infData.follower_country1.fillna('X')
    infData.follower_country2 = infData.follower_country2.fillna('X')
    infData.follower_country3 = infData.follower_country3.fillna('X')
    infData.follower_city1 = infData.follower_city1.fillna('X')
    infData.follower_city2 = infData.follower_city2.fillna('X')
    infData.follower_city3 = infData.follower_city3.fillna('X')
    infData.inf_topic1 = infData.inf_topic1.fillna('X')
    infData.inf_topic2 = infData.inf_topic2.fillna('X')
    infData.inf_topic3 = infData.inf_topic3.fillna('X')
    infData.client_topic1 = infData.client_topic1.fillna('X')
    infData.client_topic2 = infData.client_topic2.fillna('X')
    infData.client_topic3 = infData.client_topic3.fillna('X')
    infData.follower_age_bracket1 = infData.follower_age_bracket1.fillna(-1)
    infData.follower_age_bracket2 = infData.follower_age_bracket2.fillna(-1)
    infData.gender = infData.gender.fillna('X')
    infData.follower_percent_male = infData.follower_percent_male.fillna(-1)
    infData.follower_percent_female = infData.follower_percent_female.fillna(-1)
    infData.follower_percent_other = infData.follower_percent_other.fillna(-1)
    infData.age = infData.age.fillna(-1)

    del infData['price']

    hashEncoder = ce.HashingEncoder(cols=['platform', 'follower_country1', 'follower_country2', 'follower_country3', 'follower_city1', 'follower_city2', 'follower_city3', 'inf_topic1', 'inf_topic2', 'inf_topic3', 'client_topic1', 'client_topic2', 'client_topic3', 'follower_age_bracket1', 'follower_age_bracket2', 'gender'])

    encodingInputs = infData[infData.columns]

    cleanedInput = []
    count = 0

    for inpRow in request.json['input']:
        count += 1
        tempRow = []
        for inp in inpRow:
            try:
                tempRow.append(float(inp))
            except:
                tempRow.append(inp.rstrip())
        cleanedInput.append(tempRow)

    newInput = pd.DataFrame(columns=[col.rstrip() for col in request.json['columns'][0]], data=cleanedInput)

    #hashEncoder.fit(encodingInputs)
    #encoded = hashEncoder.transform(newInput)

    #encoded = hashEncoder.fit_transform(pd.concat([newInput, encodingInputs]))

    hashEncoder.fit(encodingInputs)
    encoded = hashEncoder.transform(pd.concat([newInput, encodingInputs]))

    return {"encoded": np.array2string(encoded.values[0:count], separator=',', prefix='', suffix='', formatter={'float_kind':lambda x: "%g" % x})}
