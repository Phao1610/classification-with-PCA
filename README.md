# classification with PCA using Streamlit
- Prescription Cost Analysis(PCA) 

    A method for lowering the dimensionality of such datasets, improving interpretability while minimizing information loss is principal component analysis (PCA). It does this by producing fresh, uncorrelated variables that maximize variance one after another.

  Also, it is asked, What is PCA used for?

    PCA helps in data interpretation, although it doesn’t always identify the key patterns. High-dimensional data may be made simpler via the use of principal component analysis (PCA), while still preserving trends and patterns. It does this by condensing the data into fewer dimensions that serve as feature summaries.

  Secondly, What is PCA in big data?

    A typical statistical method for reducing the dimensionality of data with a high number of connected variables is principal component analysis (PCA). When there are a lot of observations and variables, issues start to appear.

  Also, Why is PCA used in ML?

    Unsupervised statistical methods like PCA are used to condense the dataset’s dimensionality. When used with a larger input dataset, ML models with a high number of input variables or higher dimensions often fail. PCA aids in discovering connections between several variables and then coupling them.

  People also ask, Where is PCA used?

    PCA is often used in fields like face recognition, computer vision, and image compression as a dimensionality reduction approach. It is also used in the fields of finance, data mining, bioinformatics, psychology, etc. to detect patterns in high dimension data.
    
    Principal component analysis can be broken down into five steps. I’ll go through each step, providing logical explanations of what PCA is doing and simplifying mathematical concepts such as standardization, covariance, eigenvectors and eigenvalues without focusing on how to compute them.

    HOW DO YOU DO A PRINCIPAL COMPONENT ANALYSIS?
        Step 1: Standardize the range of continuous initial variables
        Step 2: Compute the covariance matrix to identify correlations
        Step 3: Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components
        Step 4: Create a feature vector to decide which principal components to keep
        Step 5: Recast the data along the principal components axes

- Streamlit Python

    Streamlit is an open-source (free) Python library, which provides a fast way to build interactive web apps. It is a relatively new package launched in 2019 but has been growing tremendously. It is designed and built especially for machine learning engineers or other data science professionals. Once you are done with your analysis in Python, you can quickly turn those scripts into web apps/tools to share with others.

    As long as you can code in Python, it should be straightforward for you to write Streamlit apps. Imagine the app as the canvas. We can write Python scripts from top to bottom to add all kinds of elements, including text, charts, widgets, tables, etc.

    In this tutorial, we’ll build an example app with Streamlit in Python. You’ll be able to use the common elements and commands from Streamlit and expand to others on your own!
    
- Demo

        Step 1: pip install streamlit
        Step 2: python -m streamlit run 20522028.py or streamlit run 20522028.py
        
    ![video demo Logistic Regression with Streamlit](https://user-images.githubusercontent.com/106755542/208859031-d67fc7f9-82a0-46e9-ad7b-f89c0a849493.mp4)
