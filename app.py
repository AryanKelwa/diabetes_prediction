from flask import Flask,render_template,request,redirect
import pickle

app=Flask(__name__)

model=pickle.load(open('model/logistic_model.pkl','rb'))
scaler=pickle.load(open('model/scaler.pkl','rb'))

@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='POST':
        Pregnancies=request.form.get('Pregnancies')
        Gluscose=request.form.get('Glucose')
        BloodPressure=request.form.get('BloodPressure')
        SkinThickness=request.form.get('SkinThickness')
        Insulin=request.form.get('Insulin')
        BMI=request.form.get('BMI')
        DiabetesPedigreeFunction=request.form.get('DiabetesPedigreeFunction')
        Age=request.form.get('Age')
        
        #standarize or scale the data
        scaled_data=scaler.transform([[Pregnancies,Gluscose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])     
        prediction=model.predict(scaled_data)
        # print([Pregnancies,Gluscose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        # print("************************\n\n")
        # print(prediction)
        
        if prediction==1:
            result="diabetic"
        else:
            result="Not diabetic"
            
        return render_template('result.htm',result=result)
    

    return render_template('index.htm')

# @app.route('/result',methods=['GET','POST'])
# def predict():
#     return render_template()

if __name__=='__main__':
    app.run(debug=True)