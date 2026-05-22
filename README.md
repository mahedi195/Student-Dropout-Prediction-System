##                                Student Dropout Prediction Web App (Django + Machine Learning )



## Deployment
deployment link

https://student-dropout-prediction-system-4.onrender.com



## Screenshorts
![image alt](https://github.com/mahedi195/Student-Dropout-Prediction-System/blob/35d6f6ff28ba5565917a125023fee20cadd589e8/home.png)


![image alt](https://github.com/mahedi195/Student-Dropout-Prediction-System/blob/d4e327e06893ec95827590e3ae867f7d7d6396d9/prediction1.png)


![image alt](https://github.com/mahedi195/Student-Dropout-Prediction-System/blob/d36517c66bead13b4e838ed24024157769fc7980/prediction2.png)



## About project
About the Project
The project uses the Kaggle Student Dropout dataset.
The dataset contains 649 rows and 34 columns.

##Dataset Columns
School, Gender, Age, Address, Family_Size, Parental_Status, Mother_Education, Father_Education, Mother_Job, Father_Job, Reason_for_Choosing_School, Guardian, Travel_Time, Study_Time, Number_of_Failures, School_Support, Family_Support, Extra_Paid_Class, Extra_Curricular_Activities, Attended_Nursery, Wants_Higher_Education, Internet_Access, In_Relationship, Family_Relationship, Free_Time, Going_Out, Weekend_Alcohol_Consumption, Weekday_Alcohol_Consumption, Health_Status, Number_of_Absences, Grade_1, Grade_2, Final_Grade, Dropped_Out


#Selected Features
For training and testing, only 8 features were selected:
[
 'Study_Time',
 'Number_of_Failures',
 'Family_Support',
 'Wants_Higher_Education',
 'Number_of_Absences',
 'Grade_2',
 'Final_Grade',
 'Dropped_Out'
]


##Train-Test Split
70% training data
30% testing data
Random state = 42
Model Used

##Model Performance
Accuracy: approximately 97%


## Tools
CSS, Python, Django


## run project 

path: (virtual) PS C:\Users\user\Desktop\student dropout\dropout\dropout> 

command: python train.py


path:(virtual) PS C:\Users\user\Desktop\student dropout\dropout> 

command: python manage.py runserver


