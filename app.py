from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/mobile_price_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs (8 features used for prediction)
        battery_power = float(request.form['battery_power'])
        ram = float(request.form['ram'])
        int_memory = float(request.form['int_memory'])
        pc = float(request.form['pc'])
        clock_speed = float(request.form['clock_speed'])
        n_cores = float(request.form['n_cores'])
        px_height = float(request.form['px_height'])
        px_width = float(request.form['px_width'])
        
        # Set default values for features not collected from user
        # These are typical/average values from the training data
        blue = 1  # Has bluetooth
        dual_sim = 1  # Has dual sim
        fc = 2  # Front camera 2MP
        four_g = 1  # Has 4G
        m_dep = 0.6  # Mobile depth 0.6cm
        mobile_wt = 140  # Mobile weight 140g
        sc_h = 12  # Screen height 12cm
        sc_w = 6  # Screen width 6cm
        talk_time = 10  # Talk time 10 hours
        three_g = 1  # Has 3G
        touch_screen = 1  # Has touch screen
        wifi = 1  # Has WiFi
        
        # Arrange features in EXACT order as training data
        features = np.array([[
            battery_power, blue, clock_speed, dual_sim, fc, four_g,
            int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
            px_width, ram, sc_h, sc_w, talk_time, three_g,
            touch_screen, wifi
        ]])
        
        # Make prediction
        prediction = int(model.predict(features)[0])
        
        # Extended labels with descriptions for "more output" and impact
        labels = {
            0: {
                'label': 'Low Cost',
                'class': 'low',
                'desc': 'This device falls into the budget-friendly category. Ideal for basic usage, calling, and light web browsing. Great value for entry-level users.'
            },
            1: {
                'label': 'Medium Cost',
                'class': 'medium',
                'desc': 'A balanced device offering good performance for the price. Suitable for daily multitasking, social media, and casual photography.'
            },
            2: {
                'label': 'High Cost',
                'class': 'high',
                'desc': 'A flagship-killer tier device. High performance with premium features, capable of handling heavy games and high-quality media consumption.'
            },
            3: {
                'label': 'Premium Cost',
                'class': 'premium',
                'desc': 'Top-tier luxury device with cutting-edge specifications. Designed for power users who demand the absolute best performance and features.'
            }
        }
        
        result_data = labels[prediction]
        
        return jsonify({
            'prediction': prediction,
            'prediction_label': result_data['label'],
            'badge_class': result_data['class'],
            'description': result_data['desc']
        })
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
