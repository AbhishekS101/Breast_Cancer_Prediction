# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from tkinter.messagebox import showinfo

root = Tk()
root.geometry('380x420')


def check(check_list):
    # Breast Cancer Detection

    # Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Importing the dataset
    dataset = pd.read_csv('dataR2_csv.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 9].values


    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size =0.25, random_state=0)


    # #my list
    # check_list = [48,23.5,70,2.707,0.467408666666667,8.8071,9.7024,7.99585,417.114]

    #adding my check_list
    check_array = np.array(check_list)
    check_array = check_array.reshape(1 , 9)
    X_train = np.append(X_train, check_array)
    X_train = X_train.reshape(88, 9)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    """sc_Y = StandardScaler()
    Y_train = sc_Y.fit_transform(Y_train)
    Y_test = sc_Y.fit_transform(Y_test)"""

    #extracting my check_list
    new_check_array = X_train[87, :]
    X_train = np.delete(X_train, 87, 0)
    new_check_array = new_check_array.reshape(1, 9)

    #Fitting Random Forest Classification Classifier to the Training Set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
    classifier.fit(X_train, Y_train)

    #predicting the Test Set Results
    Y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)


    #predicting my list
    User_pred = classifier.predict(new_check_array)

    if User_pred == 2:
        return 0
    else:
        return 1

#creating list
def entry():
    global x
    x = link.get()
    check_list = []
    check_list.append(float(link2.get()))
    check_list.append(float(link3.get()))
    check_list.append(float(link4.get()))
    check_list.append(float(link5.get()))
    check_list.append(float(link6.get()))
    check_list.append(float(link7.get()))
    check_list.append(float(link8.get()))
    check_list.append(float(link9.get()))
    check_list.append(float(link10.get()))

    healthy = 'Dear {0} \n You are healthy person and keep it up'.format(x)
    unhealthy = 'Dear {0} \n You are unhealthy person and you may have Breast Cancer soon so kindly refer to a Doctor'.format(x)
    if check(check_list) == 0:
        m = "Test Results for "
        m = m+x
        showinfo(m, healthy)
    else:
        m = "Test Results for "
        m = m+x
        showinfo(m, unhealthy)

    exit()

    # m = "Test Results for "
    # m = m+x
    # exit()

f = Frame(root)
f.grid()
Label(f, text='========BREAST CANCER DETECTOR========', font=30, padx=6).pack()
f1 = Frame(root)
f1.grid()
Label(f1, text='Enter Name here', font=5).grid(row=1)
Label(f1, text="Enter Age here", font=5).grid(row=2)
Label(f1, text="Enter BMI here", font=5).grid(row=3)
Label(f1, text="Enter Glucose here", font=5).grid(row=4)
Label(f1, text="Enter Insulin here", font=5).grid(row=5)
Label(f1, text="Enter HOMA here", font=5).grid(row=6)
Label(f1, text="Enter Leptin here", font=5).grid(row=7)
Label(f1, text="Enter Adiponectin here", font=5).grid(row=8)
Label(f1, text="Enter Resistin here", font=5).grid(row=9)
Label(f1, text="Enter MCP.1 here", font=5).grid(row=10)

link = StringVar()
link2 = StringVar()
link3 = StringVar()
link4 = StringVar()
link5 = StringVar()
link6 = StringVar()
link7 = StringVar()
link8 = StringVar()
link9 = StringVar()
link10 = StringVar()


e1 = Entry(f1, font=5, textvariable=link).grid(row=1, column=1, pady=5, padx=10)
e2 = Entry(f1, font=5, textvariable=link2).grid(row=2, column=1, pady=5, padx=10)
e3 = Entry(f1, font=5, textvariable=link3).grid(row=3, column=1, pady=5, padx=10)
e4 = Entry(f1, font=5, textvariable=link4).grid(row=4, column=1, pady=5, padx=10)
e5 = Entry(f1, font=5, textvariable=link5).grid(row=5, column=1, pady=5, padx=10)
e6 = Entry(f1, font=5, textvariable=link6).grid(row=6, column=1, pady=5, padx=10)
e7 = Entry(f1, font=5, textvariable=link7).grid(row=7, column=1, pady=5, padx=10)
e8 = Entry(f1, font=5, textvariable=link8).grid(row=8, column=1, pady=5, padx=10)
e9 = Entry(f1, font=5, textvariable=link9).grid(row=9, column=1, pady=5, padx=10)
e10 = Entry(f1, font=5, textvariable=link10).grid(row=10, column=1, pady=5, padx=10)


Button(f1, text='Initiate', padx=50, relief=RAISED, font=10, borderwidth=5, command=entry).grid(column=1, pady=5)


root.mainloop()