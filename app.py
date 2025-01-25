from flask import Flask, request, url_for, render_template
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


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if float(output) > 0.5:
        return render_template(
            'forest_fire.html',
            pred=(
                'Your Forest is in Danger.\n'
                'Probability of fire occurring is {}'.format(output)
            )
        )
    else:
        return render_template(
            'forest_fire.html',
            pred=(
                'Your Forest is safe.\n'
                'Probability of fire occurring is {}'.format(output)
            )
        )


if __name__ == '__main__':
    app.run(debug=True, port=8001)
