from flask import Flask, render_template, url_for, flash, redirect
from forms import CarsForm
import pickle
from cars_functions import calc_price

scaler_model=pickle.load(open('scaler.sav', 'rb')) 
model=pickle.load(open('adaboost.sav', 'rb')) 


app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

@app.route("/", methods=['GET', 'POST'])
def cars():
    form = CarsForm() ######CarsForm()
    if form.validate_on_submit():
        #price = calc_price(scaler_model, model, {form.cartype.data},{form.age_in_years.data},
        	#{form.performance_kw.data}, {form.speedometer.data},{form.weightTotal.data},
        	#{form.capacity.data}, {form.flag_back_w.data},{form.luggagerack.data})
        price = calc_price(scaler_model, model,'OPEL CORSA', 10000, #{form.cartype.data},{form.age_in_years.data}
        	200,10000,400,#{form.performance_kw.data}, {form.speedometer.data},{form.weightTotal.data},
        	500, 0, 599)#{form.capacity.data}, {form.flag_back_w.data},{form.luggagerack.data})
        flash('The average price of the car is {:.2f}'.format(price[0]), 'success')
    return render_template('cars.html', title='Cars', form=form)

if __name__ == '__main__':
    app.run(debug=True)
