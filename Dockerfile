FROM python:3.8.18
COPY . /covid_app
WORKDIR /covid_app
RUN pip install -r requirement.txt
CMD streamlit run covid_app.py 
