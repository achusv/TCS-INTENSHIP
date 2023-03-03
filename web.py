import  numpy as np
import pickle
from flask import Flask, render_template, request
load_model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)
@app.route('/')  
def home():
    return render_template('home.html')   
@app.route("/index")
def index():
    return render_template('index.html')
@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')
@app.route('/predict', methods=['POST','GET'])
   
def predict():
    if request.method == "POST":
        data =[]
        output=request.form
        data.append (output['age'])
        data.append (output['workclass'])
        data.append (output['education'])
        data.append (output['maritalstatus'])
        data.append (output['occupation'])
        data.append (output['relationship'])
        data.append (output['race'])
        data.append (output['sex'])
        data.append (output['hoursperweek'])
        data.append (output['nativecountry'])
        print(data)
        test=load_model.predict([data])
        print(test)
        return render_template('result.html',prediction_text='{}'.format(*test))



if __name__ == "__main__":                            
    app.run()
    
