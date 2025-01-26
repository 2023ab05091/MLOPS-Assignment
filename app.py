from flask import Flask, request, url_for, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/healthz')
def healthz():
    # Liveness probe endpoint
    # Simply returns HTTP 200 OK to indicate the server is alive
    return '', 200


@app.route('/ready')
def ready():
    # Readiness probe endpoint
    return '', 200  # Return HTTP 200 OK to indicate readiness


def redirect_url():
    return request.args.get('next') or request.referrer or url_for('index')


@app.route('/')
@app.route('/index')
def landing():
    return render_template("forest_fire.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(final)

    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    result = {}
    result['value'] = output

    if float(output) > 0.5:
        result['status'] = 'danger'
        result['text'] = 'Your Forest is in Danger. Probability of fire occurring is {}'.format(output)
        
    else:
        result['status'] = 'safe'
        result['text'] = 'Your Forest is safe. Probability of fire occurring is {}'.format(output)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8001)
