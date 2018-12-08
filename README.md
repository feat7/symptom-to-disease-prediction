# symptom-to-disease-prediction
A simple symptom to disease prediction with Machine Learning.

# How to run

1. Install the dependencies

2. Run flask app.

For linux
```
export FLASK_APP=app.py
flask run
```

For Windows
```
set FLASK_APP=app.py
python -m flask run
```

# Example API

Comma separated symptoms.

symptoms - required.

For single symptoms:
```http://localhost:5000/?symptoms=puffy_face_and_eyes```

For multiple symptoms:
```http://localhost:5000/?symptoms=puffy_face_and_eyes,pus_filled_pimples```
