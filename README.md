# Malaria_Cell_Classification

## **Introduction**

![image](https://user-images.githubusercontent.com/85283934/138677608-7c8053f3-65fd-4549-8703-ca517899b50a.png)

This Machine Learning repo is a GUI based application which uses a custom CNN model(Accuracy: 95.22%) to predict if an uploaded cell image is parasitized or uninfected. At first a Deep learning model was trained and tested in Google Colab based on the dataset obtained from kaggle, then in order to give it a user interface we use Streamlit and heroku to deploy.

## **Purpose of the Project**

Where malaria is not endemic any more (such as in the United States), health-care providers may not be familiar with the disease. Clinicians seeing a malaria patient may forget to consider malaria among the potential diagnoses and not order the needed diagnostic tests. Laboratorians may lack experience with malaria and fail to detect parasites when examining blood smears under the microscope. Malaria is an acute febrile illness.

In a non-immune individual, symptoms usually appear 10–15 days after the infective mosquito bite. The first symptoms – fever, headache, and chills – may be mild and difficult to recognize as malaria. If not treated within 24 hours, it can progress to severe illness, often leading to death.

Our Model performs fairly well with an accuracy of 95% and an F1 Score of 95% and Recall Score of 92%. This provides a handy tool to utilize the power of Machine Learning and Artificial Intelligence in Binary Classification Problems where time and accuracy is the paramount objective of classification.

## **Dataset and Description**

This is an image classification problem on Kaggle Datasets.The dataset contains 2 folders - Infected - Uninfected and has been originally taken from a government data website https://ceb.nlm.nih.gov/repositories/malaria-datasets/ .

Dataset: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

![image](https://user-images.githubusercontent.com/85283934/138679130-25d68cb0-cdee-44f9-b3d3-819d2aec1851.png)

## **Breakdown of the code**:

System will read the image uploaded by the user, augment it and will use the saved custom model to detect whether the disease is present or not in the patient and thus display the result in a user-friendly language.

Below are the steps:-

**Loading the dataset** : Load the data and import the libraries.

**Data Preprocessing** : - Reading the images,labels stored in 2 folders(Parasitized,Uninfected).

                         - Plotting the Uninfected and Parasitized images with their respective labels.
                         
                         - Normalizing the image data.
                         
                         - Train,test split
                         
                         - Data Augmentation : Augment the train and validation data using ImageDataGenerator

**Creating and Training the Model**: Create a cnn model in KERAS

**Evaluation** : Display the plots from the training history.Checked the performance using confusion matrix and classification report

**Submission**: Run predictions with model.predict, and create confusion matrix.

**Web app**: Built a web app using streamlit 



