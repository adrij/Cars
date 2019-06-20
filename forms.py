from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, IntegerField
from wtforms.validators import DataRequired


class CarsForm(FlaskForm):
    cartype = StringField('Type of the car', validators=[DataRequired()])
    age_in_years = IntegerField('Age in years',
                           validators=[DataRequired()])
    performance_kw = IntegerField('Performance in kW',
                        validators=[DataRequired()])
    speedometer = IntegerField('speedometer', validators=[DataRequired()])
    weightTotal = IntegerField('Weight', validators=[DataRequired()])
    capacity = IntegerField('Capacity', validators=[DataRequired()])
    luggagerack = IntegerField('Luggagerack', validators=[DataRequired()])
    flag_back_w = BooleanField('Back wheel prop')
    
    submit = SubmitField('Calculate')