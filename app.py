import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request

# Load the trained model
with open('asd_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    # Load the encoder object
with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)
with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)
from flask import Flask, render_template, request, redirect, session


app = Flask(__name__, template_folder='templates')
print("Testing print statement")

new_data = pd.DataFrame(columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 
                                 'A8', 'A9', 'A10', 'Age_Mons','Sex', 'Ethnicity',
                                 'Family_mem_with_ASD'])

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/preliminary1', methods=['GET', 'POST'])
def preliminary1():
    if request.method == 'POST':
        # Get form inputs
        sex = request.form['gender']
        ethnicity = request.form['ethnicity']
        birthday = datetime.strptime(request.form.get('birthday'), '%Y-%m-%d').date()
        present_date = datetime.now()
        age_in_months = (present_date.year - birthday.year) * 12 + (present_date.month - birthday.month)
        
        print("Age in months:", age_in_months)  # Debug print
        
        # Populate new_data DataFrame
        new_data.loc[0] = [None] * len(new_data.columns)
        new_data.loc[0, ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Sex', 'Ethnicity', 'Family_mem_with_ASD']] = [None, None, None, None, None, None, None, None, None, None, age_in_months, sex, ethnicity, None]
        
        # Print the new_data before preprocessing
        print("New Data (before preprocessing):")
        print(new_data)
        
        return render_template('preliminary2.html', new_data=new_data)
    
    print("Rendering preliminary1.html")  # Debug print
    return render_template('preliminary1.html')

@app.route('/preliminary2', methods=['GET', 'POST'])
def preliminary2():
    if request.method == 'POST':
        # Get form inputs
        Family_mem_with_ASD = request.form["family_diagnosed"]
        who_completed_the_test = request.form["who_completed_the_test"]
        
        # Update new_data DataFrame
        new_data.loc[0, 'Family_mem_with_ASD'] = Family_mem_with_ASD

        
        # Print the new_data before preprocessing
        print("New Data (before preprocessing):")
        print(new_data)
        
        return render_template('q1.html', new_data=new_data)
    
    print("Rendering preliminary2.html")  # Debug print
    return render_template('preliminary2.html')

# Add similar debug prints in other Flask routes as well

@app.route('/q1', methods=['GET', 'POST'])
def q1():
    if request.method == 'POST':
        A1 = request.form.get('answer1')
        
        if not A1:
            error_message = "Please select an answer for Question 1."
            return render_template('q1.html', new_data=new_data, error_message=error_message)
        
        new_data.loc[0, 'A1'] = int(A1)
        
        return render_template('q2.html', new_data=new_data)
    
    print("Rendering q1.html")  # Debug print
    return render_template('q1.html')

# Add routes and functions for the remaining questions (question2, question3, ..., question10)

@app.route('/q2', methods=['GET', 'POST'])
def q2():
    if request.method == 'POST':
        A2 = request.form.get('answer2')
        
        if not A2:
            error_message = "Please select an answer for Question 2."
            return render_template('q2.html', new_data=new_data, error_message=error_message)
        
        new_data.loc[0, 'A2'] = int(A2)
      
        return render_template('q3.html', new_data=new_data)
    
    print("Rendering q2.html")  # Debug print
    return render_template('q2.html')

@app.route('/q3', methods=['GET', 'POST'])
def q3():
    if request.method == 'POST':
        A3 = int(request.form['answer3'])
        
        new_data.loc[0, 'A3'] = A3
        return render_template('q4.html', new_data=new_data)
    print("Rendering q3.html")
    return render_template('q3.html')


@app.route('/q4', methods=['GET', 'POST'])
def q4():
    if request.method == 'POST':
        A4 = int(request.form['answer4'])
        
        new_data.loc[0, 'A4'] = A4
        return render_template('q5.html', new_data=new_data)
    print("Rendering q4.html")
    return render_template('q4.html')

@app.route('/q5', methods=['GET', 'POST'])
def q5():
    if request.method == 'POST':
        A5 = int(request.form['answer5'])
        
        new_data.loc[0, 'A5'] = A5
        return render_template('q6.html', new_data=new_data)
    print("Rendering q5.html")
    return render_template('q5.html')

@app.route('/q6', methods=['GET', 'POST'])
def q6():
    if request.method == 'POST':
       
        A6 = int(request.form['answer6'])
        
        new_data.loc[0, 'A6'] = A6
        return render_template('q7.html', new_data=new_data)
    print("Rendering q6.html")
    return render_template('q6.html')

@app.route('/q7', methods=['GET', 'POST'])
def q7():
    if request.method == 'POST':
        A7 = int(request.form['answer7'])
        
        new_data.loc[0, 'A7'] = A7
        return render_template('q8.html', new_data=new_data)
    print("Rendering q7.html")
    return render_template('q7.html')

@app.route('/q8', methods=['GET', 'POST'])
def q8():
    if request.method == 'POST':
        A8 = int(request.form['answer8'])
        
        new_data.loc[0, 'A8'] = A8

        return render_template('q9.html', new_data=new_data)
    print("Rendering q8.html")
    return render_template('q8.html')

@app.route('/q9', methods=['GET', 'POST'])
def q9():
    if request.method == 'POST':
        A9 = int(request.form['answer9'])
        
        new_data.loc[0, 'A9'] = A9
        return render_template('q10.html', new_data=new_data)
    print("Rendering q.9")
    return render_template('q9.html')

def predict(new_data):
    # Perform the prediction
    age_bins = np.arange(12, 37, 3)
    age_labels = np.arange(12, 36, 3)

    # Preprocess the new data
    new_data_encoded = pd.get_dummies(new_data, columns=['Sex', 'Ethnicity', 'Family_mem_with_ASD'])
    new_data['Age_Mons'] = new_data['Age_Mons'].astype(int)  # Convert Age_Mons to int
    print(new_data)

    # Concatenate numerical and encoded categorical features
    new_data_encoded['Age_binned'] = pd.cut(new_data['Age_Mons'], bins=age_bins.astype(int), labels=age_labels.astype(int))
    print(new_data_encoded)
    # Ensure that the new data has the same set of features as the training data
    missing_cols = set(model_columns) - set(new_data_encoded.columns)
    for col in missing_cols:
        new_data_encoded[col] = 0

    # Reorder columns to match training data
    new_data_processed = new_data_encoded[model_columns]
    print(new_data_processed)
    # Normalize feature data
    new_data_scaled = scaler.transform(new_data_processed)
    print(new_data_scaled)
    new_data_3d = new_data_scaled.reshape(new_data_scaled.shape[0], new_data_scaled.shape[1], 1)
    print(new_data_3d)
    # Make predictions on the new data
    predictions = model.predict(new_data_3d)
    print(predictions)
   


    return  (predictions)

@app.route('/q10', methods=['GET', 'POST'])
def q10():
    global new_data
    if request.method == 'POST':
        A10 = int(request.form['answer10'])
        
        new_data.loc[0, 'A10'] = A10
        print("Processing form data in q10")
        
         #Extract relevant variables from the new_data DataFrame
        sex = new_data.loc[0, 'Sex']
        ethnicity = new_data.loc[0, 'Ethnicity']
        age_in_months = new_data.loc[0, 'Age_Mons']
        family_mem_with_asd = new_data.loc[0, 'Family_mem_with_ASD']
    
        
        result = predict(new_data)
        # Render the result.html template with the prediction result
        if result < 0.5:
            res = "The child doesn't have autistic traits"
            return render_template('resultno.html', result=res, sex=sex, ethnicity=ethnicity, age_in_months=age_in_months, family_mem_with_asd=family_mem_with_asd)
        if result > 0.5:
            res = "The child has a risk of having Autism Spectrum ASD"
            return render_template('resultyes.html', result=res, sex=sex, ethnicity=ethnicity, age_in_months=age_in_months, family_mem_with_asd=family_mem_with_asd)
    
    print("Rendering q10.html")
    return render_template('q10.html')

# @app.route('/disclaimer', methods=['GET', 'POST'])
# def disclaimer():
#     if request.method == 'POST':
#           # Call the predict function to make predictions
#         result = predict(new_data)
        
#     if result < 0.5:
#         res = "The child doesn't have autistic traits"
#         return render_template('resultno.html', result=res)
#     if result > 0.5:
#         res = "The child has a risk of having Autism Spectrum ASD"
#         return render_template('resultyes.html', result=res)

@app.route('/about_asd')
def about_asd():
    return render_template('about_asd.html')
    
    
if __name__ == '__main__':
    app.run(debug=True)
    
    
  