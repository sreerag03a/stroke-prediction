from flask import  Flask, request, render_template, send_file
import numpy as np
import pandas as pd
import sys
from src.handling.logger import logging
from src.handling.exceptions import CustomException

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import io

application = Flask(__name__)

app= application

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    predpipeline = PredictPipeline()

    if request.method == 'GET':
        return render_template('predict.html')

    elif request.method == 'POST':
        f = request.files.get('file')
        model = request.form.get('model')

        if f and f.filename.strip() != '':
            logging.info('Reading file...')
            try:
                pred_df = pd.read_csv(f)
                results = predpipeline.predictstroke(pred_df,model)

                pred_df['stroke'] = results

                output = io.StringIO()
                pred_df.to_csv(output,index=False)
                output.seek(0)

                return send_file(
                    io.BytesIO(output.getvalue().encode()),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='predictions.csv'
                )

            except Exception as e:
                raise CustomException(e,sys)
        else:
            featurelist = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
            
            values_dict = {feature : request.form.get(feature) for feature in featurelist}
            data = CustomData(values_dict)
            pred_df = data.get_dataframe()
            logging.info(pred_df)
            values_dict.update({'model': model})

            try: 
                print(values_dict)
                logging.info('Prediction pipeline started')
                results = predpipeline.predictstroke(pred_df,model)
                logging.info('Prediction pipeline completed')

                print(results)
                return render_template('predict.html', results = results,prev_input = values_dict)
            
            except Exception as e:
                raise CustomException(e,sys)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)