from flask import Flask
from flask import jsonify, request

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

import string

import csv 

app=Flask(__name__)

filename = "ndd_10.txt"
 
fields = [] 
rows = [] 

with open(filename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    
    fields = csvreader.next() 

    for row in csvreader: 
        rows.append(row) 

    print("Total no. of rows: %d"%(csvreader.line_num)) 


@app.route('/')
def pred():

    results = []
    
    clf=joblib.load('FinalFile.pkl')
    for row in rows[:5]: 
        input = []
    for col in row: 
        input.append(col)
    print('input row is :', input)
    res = clf.predict(input)
    results.append(res)

    return jsonify({"results":results}),220    

if  __name__ == "__main__":
    app.run(port=8000, debug=True)
