from flask import Flask, render_template, url_for, flash, redirect, request
from forms import CarsForm
import pickle
from cars_functions import calc_price

scaler_model=pickle.load(open('scaler.sav', 'rb')) 
model=pickle.load(open('adaboost.sav', 'rb')) 


app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

@app.route("/", methods=['GET', 'POST'])
@app.route("/cars", methods=['GET', 'POST'])
def cars():
    form = CarsForm() ######CarsForm()
    if form.validate_on_submit():
        try :
        	temp=request.form['flag_back_w']
        	flag_bw=1
        except:
        	flag_bw=0
        price = calc_price(scaler_model, model, request.form['cartype'], request.form['age_in_years'], 
        	request.form['performance_kw'], request.form['speedometer'], request.form['weightTotal'],
        	request.form['capacity'], flag_bw, request.form['luggagerack'])
        
        flash('The average price of the car is {:.2f} EUR'.format(price[0]), 'success')
        #return redirect(url_for('cars'))
    return render_template('cars.html', title='Cars', form=form)

if __name__ == '__main__':
    app.run()
