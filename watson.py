import json
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
import threading
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import openpyxl as xl
import wave
import contextlib
import time

confusion = {
    "truePositive" : 0,
    "falsePositive" : 0,
    "trueNegative" : 0,
    "falseNegative" : 0
}

def metrics(confusion):
    precision = confusion["truePositive"] / (confusion["truePositive"] + confusion["falsePositive"]) * 100
    recall = confusion["truePositive"] / (confusion["truePositive"] + confusion["falseNegative"]) * 100
    f1 = 2 * (precision * recall)/(precision + recall)

    WER = ((confusion["falsePositive"] + confusion["trueNegative"] + deletions + insertions) / sentenceLength) * 100

    print("Recall = ", recall)
    print("Precision = ", precision)
    print("F1-score = ", f1)
    print("Word Error Rate = ", WER)


def compareWord(word1, word2):

    if str(word1).lower() == str(word2).lower():
        return True
    else:
        return False

wb = xl.load_workbook('transcriptionsSep.xlsx')
ws = wb['transcriptions']

sentence = []
for col in ws.iter_cols(min_row=2, min_col=2, max_col=128, max_row=2):
    for cell in col:
        if(cell.value != None):
            sentence.append(cell.value)
        else:
            break
sentenceLength = len(sentence)

authenticator = IAMAuthenticator('{APIKEY}')
service = SpeechToTextV1(authenticator=authenticator)
service.set_service_url('https://gateway-lon.watsonplatform.net/speech-to-text/api')

models = service.list_models().get_result()

model = service.get_model('en-US_BroadbandModel').get_result()
print(json.dumps(model, indent=2))

fname = './audio/Homonyms/fort_fight_vincent.wav'

with contextlib.closing(wave.open(fname,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    wavLength = frames / float(rate)

start_time = time.time()
with open(join(dirname(__file__), fname), 'rb') as audio_file:
    result = (json.dumps(
        service.recognize(
            audio=audio_file,
            content_type='audio/wav',
            word_confidence=True).get_result(),
        indent=2))
end_time = time.time()

insertions = 0
deletions = 0
i = 0
speechresult = json.loads(result)
for result in speechresult['results']:
    for alternatives in result['alternatives']:
        for words in alternatives['word_confidence']:

            if i < sentenceLength:
                if(compareWord(words[0], sentence[i]) and words[1] >= 0.50):
                    print("True positive!")
                    confusion["truePositive"] += 1

                elif (compareWord(words[0], sentence[i]) and words[1] < 0.50):
                    print("False negative!")
                    confusion["falseNegative"] += 1

                elif i < sentenceLength - 1:
                    if(compareWord(words[0], sentence[i+1])):
                        print("Word missed.. match found at i+1")
                        confusion["trueNegative"] += 1
                        deletions += 1
                        if(words[1] >= 0.50):
                            print("True positive!")
                            confusion["truePositive"] += 1    
                        else:
                            print("False negative!")
                            confusion["falseNegative"] += 1
                        i+=1

                    elif(compareWord(words[0], sentence[i-1])):
                        print("Word missed.. match found at i-1")
                        confusion["trueNegative"] += 1
                        insertions += 1
                        if(words[1] >= 0.50):
                            print("True positive!")
                            confusion["truePositive"] += 1    
                        else:
                            print("False negative!")
                            confusion["falseNegative"] += 1
                        i-=1

                    elif not compareWord(words[0], sentence[i]):
                        if words[1] >= 0.50:
                            print("False positive!")
                            confusion["falsePositive"] += 1
                        if words[1] < 0.50:
                            print("True negative!")
                            confusion["trueNegative"] += 1 


                print(words[0], "->", words[1], "    ",sentence[i], "\n")
                i += 1


print(confusion)
metrics(confusion)
print("Real time factor = ", (end_time - start_time) / wavLength)


