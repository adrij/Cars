from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, IntegerField, SelectField
from wtforms.validators import DataRequired
import pandas as pd


class CarsForm(FlaskForm):
    df_avgprice=pd.read_csv('df_avgprice.csv')
    subset = df_avgprice.iloc[:,1]
    tuples_list = list(zip(subset, subset)) 
    #tuples_list = [tuple(x) for x in subset.values]
    #n=len(subset)
    #tuples_list= [(str(x), df_avgprice.iloc[x,1]) for x in range(n)]

    cartype = SelectField('Type of the car', choices=tuples_list)
    age_in_years = IntegerField('Age in years',
                           validators=[DataRequired()])
    performance_kw = IntegerField('Performance in kW',
                        validators=[DataRequired()])
    speedometer = IntegerField('speedometer', validators=[DataRequired()])
    weightTotal = IntegerField('Weight total', validators=[DataRequired()])
    capacity = IntegerField('Capacity', validators=[DataRequired()])
    luggagerack = IntegerField('Luggagerack', validators=[DataRequired()])
    flag_back_w = BooleanField('Back wheel drive')
    
    submit = SubmitField('Calculate')