from distutils.command.upload import upload
from tkinter import Button
from PIL import Image
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import pickle
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



st.markdown(
"""
# Nguyễn Văn Toàn
# Mssv: 20522028
## Principal Components Regression với Streamlit 
""")

uploaded_file = st.file_uploader("Dataset")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open('./'+uploaded_file.name, "wb") as f: 
        f.write(bytes_data)
    file = pd.read_csv(uploaded_file)
    # st.write(file.head())
    if st.checkbox('View dataset in table data format'):
        st.dataframe(file)
    # header = file_clear.columns.values
    # Input_feature = st.selectbox(" ", ("Principal Component Analysis", "No Principal Component Analysis"), label_visibility="collapsed")

    # if Input_feature == "Principal Component Analysis":
    #     # st.dataframe(X1)
    #     # from sklearn.preprocessing import StandardScaler
    #     # scaler = StandardScaler()
    #     # scaler.fit(file_clear)
    #     # scaled_data = scaler.transform(file_clear)
    #     amount = st.selectbox(" ", range(1, file_clear.shape[1]+1), label_visibility="collapsed")
    #     pca = PCA(n_components=amount)
    #     pca.fit(file_clear)
    #     file = pca.transform(file_clear)
    # else:
    #     file = file_clear

    header = file.columns.values
    st.write("file name",file)
    X = pd.DataFrame()
    input_features = []
    dem = 0
    st.header("Input Features")
    cols = st.columns(4)
    for i in range(len(header)):
        cbox = cols[int(i/len(header)*4)].checkbox(header[i])
        if cbox:
            input_features.append(header[i])
            X.insert(dem,dem,file[header[i]])
            dem = dem + 1
    st.dataframe(X)
    options = [header for header in header if header not in input_features and file.dtypes[header] != 'object']
    st.header("Type of Splitting Data")
    
    # Input_feature = st.selectbox(" ", ("Principal Component Analysis", "No Principal Component Analysis"), label_visibility="collapsed")

    # split_type = st.selectbox(" ", ("Train-Test Split", "K-Fold Cross Validation"), label_visibility="collapsed")
    cols = st.columns(2)
    # with cols[0]:
        # st.header("Output Feature")
    output_feature = st.radio("Output Feature",options)
    st.write(file[output_feature].values)
    y = file[output_feature]
    # st.write(X.shape,' ',y.shape)
    encs = []
    Y = file[output_feature].to_numpy()
    XX = np.array([])
    enc_idx = -1  
    for feature in input_features:
        x = file[feature].to_numpy().reshape(-1, 1)
        if (file.dtypes[feature] == 'object'):
            encs.append(OneHotEncoder(handle_unknown='ignore'))
            enc_idx += 1
            x = encs[enc_idx].fit_transform(x).toarray()
        if len(XX)==0:
            XX = x
        else:
            XX = np.concatenate((XX, x), axis=1)
    Input_feature = st.selectbox(" ", ("Principal Component Analysis", "No Principal Component Analysis"))
            
    split_type = st.selectbox(" ", ("Train-Test Split", "K-Fold Cross Validation"))

    if split_type == "Train-Test Split":
        st.subheader("Train/Test Split")
        train = st.slider("Train/Test Split",0,10,1)
        k = train / 10
        X_train, X_test, y_train, y_test = train_test_split(XX, Y, train_size=k, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        if Input_feature == "Principal Component Analysis":
            k_pca = st.selectbox(" ", range(1, X_train.shape[1]+1))
            pca = PCA(n_components = k_pca)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test) 
            # explained_variance = pca.explained_variance_ratio_?

        Run = st.button("Run")
        if Run:   
            model = LogisticRegression()
            model_fit = model.fit(X_train, y_train)
            y_pred = model_fit.predict(X_test)
            # matrix = confusion_matrix(Y_train, model.predict(X_train))
            # st.write(matrix)

            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)
            score = accuracy_score(y_test, y_pred)
            st.write("Accuracy Score =", score)
            # fig_train = plt.figure(figsize=(8, 8))
            # plt.subplot(211)
            # X_set, y_set = sc.inverse_transform(X_train), y_train
            # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
            #                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
            # plt.contourf(X1, X2, model.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
            #              alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
            # plt.xlim(X1.min(), X1.max())
            # plt.ylim(X2.min(), X2.max())
            # for i, j in enumerate(np.unique(y_set)):
            #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
            # plt.title('Logistic Regression (Training set)')
            # plt.xlabel('Age')
            # plt.ylabel('Estimated Salary')
            # # plt.legend()
            # # plt.show()
            # st.pyplot(fig_train)

            # fig_test = plt.figure(figsize=(8, 8))
            # plt.subplot(211)
            # X_set, y_set = sc.inverse_transform(X_test), y_test
            # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
            #                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
            # plt.contourf(X1, X2, model.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
            #              alpha = 0.75, cmap = ListedColormap(('salmon', 'dodgerblue')))
            # plt.xlim(X1.min(), X1.max())
            # plt.ylim(X2.min(), X2.max())
            # for i, j in enumerate(np.unique(y_set)):
            #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('salmon', 'dodgerblue'))(i), label = j)
            # plt.title('Logistic Regression (Testing set)')
            # plt.xlabel('Age')
            # plt.ylabel('Estimated Salary')
            # # plt.legend()
            # # plt.show()
            # st.pyplot(fig_test)
    else:
        st.header("Numbers of Fold")
        k_fold = st.selectbox(" ", range(2, X.shape[0]))
        # Run = st.button("Run")
        kf = KFold(n_splits=k_fold, random_state=None)
        folds = [str(fold) for fold in range(1, k_fold+1)]
        score = []
        for train_index, test_index in kf.split(XX):
            X_train, X_test = XX[train_index, :], XX[test_index, :]
            Y_train, Y_test = Y[train_index], Y[test_index]
        if Input_feature == "Principal Component Analysis":
            k_pca = st.selectbox(" ", range(1, X_train.shape[1]+1))
            pca = PCA(n_components = k_pca)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        Run = st.button("Run")
        if Run: 
            model = LogisticRegression().fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            score.append(round(accuracy_score(y_true=Y_test, y_pred=Y_pred), 2))
            with open('model.pkl','wb') as f:
                pickle.dump(model, f)
            # st.write(score)
            img_score = plt.figure(figsize=(8, 8))
            ax1 = plt.subplot()
            ax1.bar(np.arange(len(folds)) - 0.17, score, 0.5, label='SCORE', color='blue')
            plt.xticks(np.arange(len(folds)), folds)
            plt.xlabel("Folds", color='blue')
            # plt.ylabel("Accuracy Score", color='maroon')
            plt.title("Accuracy Score")
            st.pyplot(img_score)
            # plt.savefig('chart.png')
 
