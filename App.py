from flask import Flask, render_template, request, session, flash

import mysql.connector
import base64, os

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from requests import get
from bs4 import BeautifulSoup
import os
from flask import Flask, render_template, request, jsonify

english_bot = ChatBot('Bot',
                      storage_adapter='chatterbot.storage.SQLStorageAdapter',
                      logic_adapters=[
                          {
                              'import_path': 'chatterbot.logic.BestMatch'
                          },

                      ],
                      trainer='chatterbot.trainers.ListTrainer')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'aaa'



@app.route("/ask", methods=['GET', 'POST'])
def ask():
    message = str(request.form['messageText'])
    bott = ''
    bott1 = ''
    sresult1 = ''

    bot_response = english_bot.get_response(message)

    print(bot_response)

    while True:

        if bot_response.confidence > 0.5:

            bot_response = str(bot_response)
            print(bot_response)
            return jsonify({'status': 'OK', 'answer': bot_response})

        elif message == ("bye") or message == ("exit"):

            bot_response = 'Hope to see you soon' + '<a href="http://127.0.0.1:5000">Exit</a>'

            print(bot_response)
            return jsonify({'status': 'OK', 'answer': bot_response})
            break

        else:

            try:
                url = "https://en.wikipedia.org/wiki/" + message
                page = get(url).text
                soup = BeautifulSoup(page, "html.parser")
                p = soup.find_all("p")
                return jsonify({'status': 'OK', 'answer': p[1].text})



            except IndexError as error:

                bot_response = 'Sorry i have no idea about that.'

                print(bot_response)
                return jsonify({'status': 'OK', 'answer': bot_response})

@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/chat")
def chat():
    return render_template('chat.html')


@app.route('/AdminLogin')
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route('/UserLogin')
def UserLogin():
    return render_template('UserLogin.html')


@app.route('/NewUser')
def NewUser():
    return render_template('NewUser.html')


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':
        if request.form['uname'] == 'admin' and request.form['password'] == 'admin':

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1skincancerdb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb ")
            data = cur.fetchall()
            flash("you are successfully Login")
            return render_template('AdminHome.html', data=data)

        else:
            flash("UserName or Password Incorrect!")
            return render_template('AdminLogin.html')


@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1skincancerdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb  ")
    data = cur.fetchall()
    return render_template('AdminHome.html', data=data)


@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':
        name = request.form['name']
        mobile = request.form['mobile']
        email = request.form['email']
        address = request.form['address']
        username = request.form['uname']
        password = request.form['password']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1skincancerdb')
        cursor = conn.cursor()
        cursor.execute(
            "insert into regtb values('','" + name + "','" + mobile + "','" + email + "','" + address + "','" + username + "','" + password + "')")
        conn.commit()
        conn.close()
        flash("Record Saved!")

    return render_template('NewUser.html')


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['sname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1skincancerdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and password='" + password + "'")
        data = cursor.fetchone()
        if data is None:
            flash('Username or Password is wrong')
            return render_template('UserLogin.html', data=data)

        else:
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1skincancerdb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where username='" + username + "' and password='" + password + "'")
            data = cur.fetchall()
            flash("you are successfully logged in")
            return render_template('UserHome.html', data=data)


@app.route('/UserHome')
def UserHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1skincancerdb')
    cur = conn.cursor()
    cur.execute("SELECT username FROM regtb  where username='" + session['sname'] + "' ")
    data = cur.fetchall()
    return render_template('DoctorHome.html', data=data)


@app.route('/Predict')
def Predict():
    return render_template('Predict.html')


@app.route("/imupload", methods=['GET', 'POST'])
def imupload():
    if request.method == 'POST':
        import cv2
        file = request.files['file']
        file.save('static/Out/Test.jpg')

        import_file_path = 'static/Out/Test.jpg'

        import tensorflow as tf
        classifierLoad = tf.keras.models.load_model('Model/skinmodel.h5')

        import numpy as np
        from keras.preprocessing import image

        test_image = image.load_img('static/Out/Test.jpg', target_size=(200, 200))
        img1 = cv2.imread('static/Out/Test.jpg')
        # test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)

        out = ''
        pre = ''
        if result[0][0] == 1:
            print("BasalCellCarcinoma")
            out = "BasalCellCarcinoma"
            pre = "5-fluorouracil (5-FU)"
        elif result[0][1] == 1:
            print("CutaneousT-celllymphoma")
            out = "CutaneousT-celllymphoma"
            pre = "Bone marrow transplant"
        elif result[0][2] == 1:
            print("DermatofibrosarcomaProtuberans")
            out = "DermatofibrosarcomaProtuberans"
            pre = "Radiation therapy"
        elif result[0][3] == 1:
            print("KaposiSarcoma")
            out = "KaposiSarcoma"
            pre = "Antiretroviral therapy for HIV also treats KS"

        elif result[0][4] == 1:
            print("MerkelCellcarCinoma")
            out = "MerkelCellcarCinoma"
            pre = "Immunotherapy"
        elif result[0][5] == 1:
            print("SebaceousGlandCarcinoma")
            out = "SebaceousGlandCarcinoma"
            pre = "dequate surgical excision"
        elif result[0][6] == 1:
            print("SquamousCellCarcinoma")
            out = "SquamousCellCarcinoma"
            pre = "Photodynamic therapy"

        return render_template('Predict.html', res=out, pre=pre)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
