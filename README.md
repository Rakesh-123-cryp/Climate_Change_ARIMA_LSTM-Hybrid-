# 7edbb52a-ffe1-48cd-b5a5-fecdcec3f5c7
https://sonarcloud.io/summary/overall?id=examly-test_7edbb52a-ffe1-48cd-b5a5-fecdcec3f5c7

NASSCOM academic grand challenge on climate change

Team name: HARM

# Description

The datasets we used are present in the Datasets/ and Datasets1/ folders.
The Heat_predictions.py file contains the python script used to generate predictions for the temperature of all the weeks and months in the year 2023 for the cities Warangal, Nizamabad, Karimnagar, Khammam, Adilabad.
The AQI_predictions.py file contains the python script used to generate predictions for the Air Quality Index of all the weeks and months in the year 2023 for the cities Warangal, Nizamabad, Karimnagar, Khammam, Adilabad.

Due to RAM constraints, the VSCode workbench provided was unable to train the LSTM models, and thus the predictions for heat waves and Air quality index are provided seperately in the Heat_wave_predictions.pdf and the AQI_predictions.pdf files respectfully.

The app.py file contains the python FLASK script for the frontend website [found here](https://8080-eeadbadeabbaefdfedabceacf.examlyiopb.examly.io/)
python3 app.py 
runs the website.

The html templates are found in the templates/ folder. 

# Requirements

Run the command given below in the terminal to install the required dependencies.

pip install -r requirements.txt
