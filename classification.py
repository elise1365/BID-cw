import pandas as pd
import scipy.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# opens the testing labels and converts then to a dataframe
def openTestLabels():
    path = r"C:\Users\pokem\OneDrive\Documents\BID cw datasets\LabelTestAll.mat"

    # open matlab file
    labels = scipy.io.loadmat(path)
    
    # Take first item from labelTest - contains all the data, just formatted badly
    labels_test = labels['LabelTest'][0]
    
    # Time to tidy up data:
    # Create a list to store the data
    lst = []
    
    # Iterate over each tuple in the label_train array
    for arr in labels_test:
        # Access the elements of the array
        imgName = arr[0].tolist()[0]
        label = arr[1].tolist()
        # label data is 1st item in a dictionary, again, just formatted weirdly
        label = label[0]
        
        # Take the labels and convert them to the named columns from the readme
        x,y,w,h,face_type,x1,y1,w1,h1, occ_type, occ_degree, gender, race, orientation, x2,y2,w2,h2 = label
        
        lst.append({'imgName': imgName, 'x':x, 'y':y, 'w':w, 'h':h, 
                          'face_type':face_type, 'x1':x1, 'y1':y1, 'w1':w1, 'h1':h1,
                          'occ_type':occ_type, 'occ_degree':occ_degree, 'gender':gender,
                          'race':race, 'orientation':orientation, 'x2':x2, 'y2':y2,
                          'w2':w2, 'h2':h2
                          })
    # convert to dataframe
    testDf = pd.DataFrame(lst)
    
    return testDf
    
# opens the training labels and converts then to a dataframe
def openTrainLabels():
    # Opens the LabelTestAll.mat file -> this is the labels for the test images
    path = r"C:\Users\pokem\OneDrive\Documents\BID cw datasets\LabelTrainAll.mat"

    labels = scipy.io.loadmat(path)

    # Extract the data from the loaded dictionary
    label_train = labels['label_train'][0]

    # Create a list to store the data
    lst = []

    # Iterate over each tuple in the label_train array
    for arr in label_train:
        # Access the elements of the array
        orgImgName = arr[0].tolist()[0]
        imgName = arr[1].tolist()[0]
        label = arr[2].tolist()
        label = label[0]
        
        # Take the labels and convert them to the named columns from the readme
        x, y, w, h, x1, y1, x2, y2, x3, y3, w3, h3, occ_type, occ_degree, gender, race, orientation, x4, y4, w4, h4 = label
             
        lst.append({'orgImgName': orgImgName, 'imgName': imgName, 
                          'x': x, 'y': y, 'w': w, 'h': h,
                          'x1': x1, 'y1': y1,
                          'x2': x2, 'y2': y2,
                          'x3': x3, 'y3': y3, 'w3': w3, 'h3': h3,
                          'occ_type': occ_type, 'occ_degree': occ_degree,
                          'gender': gender, 'race': race,
                          'orientation': orientation,
                          'x4': x4, 'y4': y4, 'w4': w4, 'h4': h4})
    
    # Create a Pandas DataFrame from the list of dictionaries
    trainDf = pd.DataFrame(lst)
    
    return trainDf

def randomForestModel():
    # Because the labels for the training and testing dataset are slightly different, select the features they share
    features = ['x', 'y', 'w', 'h', 'occ_type', 'occ_degree', 'gender', 'race', 'orientation']
    
    train_df = openTrainLabels()
    x_train = train_df[features].values
    #y_train is expected output, the category we are training the model on
    y_train = (train_df['occ_type'] != 3).values.astype(int)
    
    test_df = openTestLabels()
    test_df = test_df[test_df['face_type'].notna()] 
    x_test = test_df[features].values
    y_test = (test_df['face_type'] == 1).values.astype(int)
    
    # random forest classifier trained with the training sets
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(x_train, y_train)
    
    # predict whether each item in x_test contains a face and compare to y_train (indicates whether each is a face)
    y_pred = rf_classifier.predict(x_test)
    
    evaluateResults(y_test, y_pred)

# Results analysis and visualisation
def evaluateResults(test, predictions):
    # Accuracy
    accuracy = accuracy_score(test, predictions)
    accuracy = accuracy * 100
    print(f"Accuracy on testing data: {accuracy} %")
    # Confusion matrix
    conf_matrix = confusion_matrix(test, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    # Precision, recall & f1 score
    print("Classification Report:")
    print(classification_report(test,predictions))

randomForestModel()
