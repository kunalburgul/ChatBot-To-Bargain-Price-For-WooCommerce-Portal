
import pickle
import numpy as np
from flask import Flask, render_template, url_for, request
from werkzeug.utils import redirect


app = Flask(__name__)


@app.route('/')
def search():
    return render_template('index.html')


def valuepredictor(to_predict_list):
    to_predict = np.array(to_predict_list[0:3]).reshape(1, 3)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


count = 0


@app.route('/result', methods=['GET', 'POST'])
def results():
    global count

    if request.method == 'GET':
        return redirect(url_for('/'))
    else:
        actualprice = request.form['ActualPrice']
        userprice = request.form['UserPrice']
        customerid = request.form['CustomerID']
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = valuepredictor(to_predict_list)
        if count == 0:
            disprice = (int(actualprice) * ((result/100)/2))
            finalprice = (int(actualprice)-disprice)
            if float(finalprice) < float(userprice):
                finalprice = userprice

            userprice_list = []
            file = open('data.txt', 'a')
            file.write("User:")
            file.writelines(userprice)
            file.write('\n')
            file.write("System:")
            file.writelines(str(int(finalprice)))
            file.write('\n')
            file.close()
            file = open('data.txt', 'r')
            for line in open('data.txt'):
                userprice_list.append(file.readlines())
            count += 1
            global prevcustomerID
            prevcustomerID = customerid
            return render_template('result.html', userprice_list=userprice_list, userprice=userprice, prediction=result,
                                   discountedprice=finalprice)
        elif count == 1:
            disprice = (int(actualprice) * ((result / 100)*(3/4)))
            finalprice = (int(actualprice) - disprice)
            if float(finalprice) < float(userprice):
                finalprice = userprice

            userprice_list = []
            file = open('data.txt', 'a')
            file.write("User:")
            file.writelines(userprice)
            file.write('\n')
            file.write("System:")
            file.writelines(str(int(finalprice)))
            file.write('\n')
            file.close()
            file = open('data.txt', 'r')
            for line in open('data.txt'):
                userprice_list.append(file.readlines())
            count += 1
            global prevcustomerID1
            prevcustomerID1 = prevcustomerID
            return render_template('result1.html', userprice_list=userprice_list, userprice=userprice, prediction=result,
                                   discountedprice=finalprice)
        else:
            disprice = (int(actualprice) * (result / 100))
            finalprice = (int(actualprice) - disprice)
            if float(finalprice) < float(userprice):
                finalprice = userprice

            userprice_list = []
            file = open('data.txt', 'a')
            file.write("User:")
            file.writelines(userprice)
            file.write('\n')
            file.write("System:")
            file.writelines(str(int(finalprice)))
            file.write('\n')
            file.close()
            file = open('data.txt', 'r')
            for line in open('data.txt'):
                userprice_list.append(file.readlines())
            count += 1
            return render_template('result2.html', userprice_list=userprice_list, userprice=userprice,
                                   prediction=result,
                                   discountedprice=finalprice)


if __name__ == '__main__':
    app.run(port=5023, debug=True)
