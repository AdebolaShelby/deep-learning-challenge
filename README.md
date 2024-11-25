# Deep Learning Challenge

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Using machine learning and neural networks, I used the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Data Structure

From Alphabet Soup’s business team, we have a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

    * EIN and NAME—Identification columns

    * APPLICATION_TYPE—Alphabet Soup application type

    * AFFILIATION—Affiliated sector of industry

    * CLASSIFICATION—Government organization classification

    * USE_CASE—Use case for funding

    * ORGANIZATION—Organization type

    * STATUS—Active status

    * INCOME_AMT—Income classification

    * SPECIAL_CONSIDERATIONS—Special considerations for application

    * ASK_AMT—Funding amount requested

    * IS_SUCCESSFUL—Was the money used effectively

# Execution

## Step 1: Preprocess the Data:

    * Read the charity_data.csv file into a Pandas DataFrame.

    * Drop non-beneficial ID columns (EIN and NAME).

    * Identify and replace rare categorical variables in APPLICATION_TYPE and CLASSIFICATION columns with "Other".

    * Encode categorical variables using pd.get_dummies().

    * Split the preprocessed data into features (X) and target (y) arrays.

    * Split the data into training and testing datasets.

    * Scale the feature data using StandardScaler.



## Compile, Train, and Evaluate the Model:

    * Define a deep neural network model using TensorFlow and Keras.

    * Add multiple hidden layers with appropriate activation functions.

    * Add dropout layers to prevent overfitting.

    * Compile the model with binary_crossentropy loss function and adam optimizer.

    * Train the model with early stopping callback to prevent overfitting.

    * Evaluate the model using the test data to determine the loss and accuracy.


## Save the Model:

    * Export the trained model to an HDF5 file named AlphabetSoupCharity.h5.


# Result Summary

## Model Training:

    * A neural network model was trained using the preprocessed charity dataset.
    * The model architecture included multiple hidden layers with ReLU activation functions and dropout layers to mitigate overfitting.
    * Early stopping was employed during training to prevent overfitting.

## Model Evaluation:

    * The model was evaluated using the test dataset.

    * The final accuracy score achieved was approximately 77.81%, surpassing the target accuracy of 75%.


## Model Export:

The trained model was successfully saved to an HDF5 file named AlphabetSoupCharity.h5.

Overall, after various rounds of optimizations, the model performed well, achieving an accuracy higher than the desired 75%, and was saved for future use or further optimization.