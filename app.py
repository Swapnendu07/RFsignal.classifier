from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('radio_signal_model.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Check for file upload first
            if 'file' in request.files and request.files['file'].filename != '':
                uploaded_file = request.files['file']
                contents = uploaded_file.read().decode('utf-8')
                iq_data = contents.strip()
            else:
                iq_data = request.form['iq_data']

            iq_list = [float(x.strip()) for x in iq_data.split(',')]
            if len(iq_list) != 2048:
                prediction = "Please enter exactly 2048 float values."
            else:
                # Reshape into (1, 1024, 2, 1)
                sample = np.array(iq_list).reshape(
                    1, 1024, 2, 1).astype('float32')
                pred = model.predict(sample)
                predicted_label = encoder.inverse_transform([np.argmax(pred)])[
                    0]
                prediction = f"Predicted Class: {predicted_label}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
