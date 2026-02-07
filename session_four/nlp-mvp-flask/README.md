# NLP MVP — Flask demo

Tez boshlash:

1. Virtual env yarating va o‘rnatish:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt


2. Modelni shugullantirish
python model/train_model.py --out model/classifier.joblib

3. Flaskni ishga tushurish
python app.py (yoki flask run)
