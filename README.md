# fake_news_detection 

This application gathers tweets, predicts their category between **Real**, **Fake** and then shows a summary.

  *The scraped tweets are converted into a numeric feature vector with TF-IDF vectorization

  *Then, a Support Vector Classifier is applied to predict each category. 

  *Finally, the results are visualized in a graph and a pie chart.

-->press the **Scrape** button.
# Installation :
Clone this repo to your local machine using https://github.com/nahouda/fake_news_detection.git

conda create env -f environment 

conda activate dash

pip install -r requirements.txt

python test.py

# Software and Library Requirements:

requirements.txt:
attrs==19.1.0
beautifulsoup4==4.7.1
certifi==2019.6.16
chardet==3.0.4
Click==7.0
dash==1.0.1
dash-core-components==1.0.0
dash-html-components==1.0.0
dash-renderer==1.0.0
dash-table==4.0.1
decorator==4.4.0
Flask==1.1.1
Flask-Compress==1.4.0
gunicorn==19.9.0
idna==2.8
ipython-genutils==0.2.0
itsdangerous==1.1.0
Jinja2==2.10.1
joblib==0.13.2
jsonschema==3.0.1
jupyter-core==4.5.0
lxml==4.3.4
MarkupSafe==1.1.1
nbformat==4.4.0
nltk==3.4.4
numpy==1.16.4
pandas==0.24.2
plotly==3.10.0
pyrsistent==0.15.3
python-dateutil==2.8.0
pytz==2019.1
PyYAML==5.1.1
requests==2.22.0
retrying==1.3.3
scikit-learn==0.21.2
scipy==1.3.0
six==1.12.0
soupsieve==1.9.2
traitlets==4.3.2
urllib3==1.25.3
Werkzeug==0.15.4
wincertstore==0.2



