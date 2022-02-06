from flask import Flask, app, render_template, Markup, request
import csv
import numpy as np
import pickle
import os


app = Flask(__name__)

# get root path for account in cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(BASE_DIR + '/model.pkl', 'rb'))

# survey page
@app.route("/", methods=['POST', 'GET'])
def survey_page():
    message = ''
    if request.method == 'POST':
        questionList = []
        for n in range(1, 43):
            if f"question{n}" in request.form:
                questionList.append(request.form[f"question{n}"])
            else:
                questionList.append('')

        # # check that essential fields have been filled
        message = ''
        missing_required_answers_list = []

        if len(questionList) < 42:
            missing_required_answers_list.append(1)


        if len(missing_required_answers_list) > 0:
            # return back a string with missing fields
            message = '<div class="w3-row-padding w3-padding-16 w3-center"><H3>You missed the following question(s):</H3><font style="color:red;">'
            for ms in missing_required_answers_list:
                message += '<BR>' + str(ms)
            message += '</font></div>'
        else:
            # save to file and send thank you note
            with open(BASE_DIR + '/surveys/survey_samp_1.csv','w+') as myfile: # use a+ to append and create file if it doesn't exist
                questionString = ",".join(questionList)
                #print(questionString)
                myfile.write(
                    str(questionString) + '\n')

            message = '<div class="w3-row-padding w3-padding-16 w3-center"><H2><font style="color:blue;">Thank you for taking the time to complete this survey!</font></H2></div>'

    return render_template('survey.html',message = Markup(message))


@app.route('/predict')
def predict():
    with open(BASE_DIR + "/surveys/survey_samp_1.csv", 'r') as csvfile:
        df = csv.reader(csvfile)
        d = list(next(df))
        data = np.array(d)
    data = data.reshape(1, -1)
    prediction = model.predict(data)
    output = prediction[0]
    anxiety = ""
    if output == 1:
        anxiety = "You have anxiety"
    else:
        anxiety = "You don't have anxiety"
    message = f'<div class="w3-row-padding w3-padding-16 w3-center"><H2><font style="color:blue;">Thank you for taking the time to complete this survey!\n{anxiety}</font></H2></div>'


    return render_template('result.html',prediction_text=Markup(message))
