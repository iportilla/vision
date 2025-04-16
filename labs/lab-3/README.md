13.93.149.67

http://13.93.149.67:5000/

http://13.93.149.67:5000/app/

curl -F "image=@./450px-Broadway_and_Times_Square_by_night.jpg" -X POST http://13.93.149.67:5000/model/predict


nyc-street.jpg

curl -F "image=@./nyc-street.jpg" -X POST http://13.93.149.67:5000/model/predict
