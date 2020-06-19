import ktrain

#model location to be changed to where the model has been downloaded
#model_location = '../models/Sentiment Analysis-IMDb-BERT/predictor'
model_location = 'C:\GIT\Sentiment-Analysis-BERT-KTrain\models\Sentiment Analysis-IMDb-BERT\predictor'
predictor = ktrain.load_predictor(model_location)

def returnSentiment(query):
  probab = predictor.predict(query, return_proba=True)
  results = dict()
  results ['neg']  = str(probab[0])
  results['pos'] = str(probab[1])
  return results
