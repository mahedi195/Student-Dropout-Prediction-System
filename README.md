##                                                     Student Dropout Prediction System

## Screenshorts
![image alt](https://github.com/mahedi195/Student-Dropout-Prediction-System/blob/35d6f6ff28ba5565917a125023fee20cadd589e8/home.png)


![image alt](https://github.com/mahedi195/Student-Dropout-Prediction-System/blob/d4e327e06893ec95827590e3ae867f7d7d6396d9/prediction1.png)


![image alt](https://github.com/mahedi195/Student-Dropout-Prediction-System/blob/d36517c66bead13b4e838ed24024157769fc7980/prediction2.png)




## About project

-kaggle student dropout dtaset used

-there are 649 rows × 34 columns in kaggle dataset

-columns of kaggle dtaset -> 'School'  'Gender' 'Age' 'Address' 'Family_Size' 'Parental_Status' 'Mother_Education' 'Father_Education' 'Mother_Job' 'Father_Job' 'Reason_for_Choosing_School' 'Guardian' 'Travel_Time' 'Study_Time' 'Number_of_Failures'  'School_Support'  'Family_Support' 'Extra_Paid_Class'
'Extra_Curricular_Activities'  'Attended_Nursery'  'Wants_Higher_Education' 'Internet_Access'
'In_Relationship' 'Family_Relationship' 'Free_Time' 'Going_Out' 'Weekend_Alcohol_Consumption' 'Weekday_Alcohol_Consumption' 'Health_Status' 'Number_of_Absences' 'Grade_1' 'Grade_2' 'Final_Grade' 'Dropped_Out'


-BUT, we take only 8 (eight) features for testing and training --->['Study_Time', 'Number_of_Failures', 'Family_Support', 'Wants_Higher_Education', 'Number_of_Absences', 'Grade_2', 'Final_Grade', 'Dropped_Out']


-Tesing 30% data using 42 random state


-Artificial Neural Network (ANN) Model is used with 32,26,8 nurons 


-Accuracy ~97% 




## Tech stack

CSS, PYTHON, Django



## Deployment
deployment link

https://student-dropout-prediction-system-4.onrender.com


## run project 

path: (virtual) PS C:\Users\user\Desktop\student dropout\dropout\dropout> 

command: python train.py


path:(virtual) PS C:\Users\user\Desktop\student dropout\dropout> 

command: python manage.py runserver


