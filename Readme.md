# Spam SMS Detector Web Application

This project is a web application that uses a machine learning model to predict whether an SMS message is spam or ham (not spam). The application is built using Python, Flask (a micro web framework), and scikit-learn for the machine learning model.

## Project Structure
```bash
    Spam_Mail/
    ├── app.py             # Main Flask application file
    ├── templates/         # Directory for HTML templates
    │   └── index.html     # The main web page
    ├── static/            # Directory for static files (CSS)
    │   └── style.css      # CSS file for styling
    ├── text_preprocessor.joblib   # Saved text preprocessor model
    ├── feature_engineer.joblib    # Saved feature engineering model
    ├── tfidf_vectorizer.joblib    # Saved TF-IDF vectorizer
    ├── spam_classifier.joblib     # Saved trained classification model
    ├── spam.csv             # The original dataset (for training)
    └── README.md            # This file    
```

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone [<repository_url>](https://github.com/Abhay-hack/Spam-Mail.git)
    cd Spam_Mail
    ```

2.  **Install required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file if you haven't already. You can generate it using `pip freeze > requirements.txt` after installing the necessary libraries: `pandas`, `nltk`, `scikit-learn`, `imblearn`, `matplotlib`, `seaborn`, `joblib`, `flask`)*

3.  **Download NLTK resources:**
    The script requires some NLTK (Natural Language Toolkit) resources. Run the `spam.py` script once to download them:
    ```bash
    python spam.py
    ```
    This will download `punkt`, `stopwords`, and `wordnet` if they are not already present.

## Running the Application

1.  **Train and Save the Model (if not already done):**
    If you haven't trained and saved the model components (`text_preprocessor.joblib`, `feature_engineer.joblib`, `tfidf_vectorizer.joblib`, `spam_classifier.joblib`), run the `spam.py` script:
    ```bash
    python spam.py
    ```
    This script will load the data, preprocess it, train a logistic regression model with feature engineering, evaluate it, save the trained components, and then start the Flask development server.

2.  **Access the Web Application:**
    Once the Flask development server is running (you should see output indicating the server address, usually `http://127.0.0.1:5000/`), open your web browser and navigate to that address.

## Using the Application

1.  **Enter SMS Message:** On the web page, you will find a text area labeled "Enter SMS Message:". Type or paste the SMS message you want to classify into this area.

2.  **Analyze Message:** Click the "Analyze Message" button.

3.  **View Prediction:** The application will process the text using the loaded machine learning model and display the prediction ("Spam" or "Ham") along with a confidence score below the input form.

## Model Details

The spam detection model uses the following techniques:

* **Text Preprocessing:** Lowercasing, URL removal, punctuation removal (except '/'), stop word removal, and lemmatization.
* **Feature Engineering:** Extraction of text length and presence of capitalized words.
* **Text Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors.
* **Model:** Logistic Regression, trained on a dataset of labeled SMS messages.
* **Handling Class Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique) is used to address the imbalance between spam and ham messages in the training data.

## Libraries Used

* **pandas:** For data manipulation and analysis.
* **nltk:** For natural language processing tasks (tokenization, stopwords, lemmatization).
* **scikit-learn:** For machine learning algorithms (TfidfVectorizer, LogisticRegression, train\_test\_split, metrics, Pipeline), and base classes for custom transformers.
* **imblearn:** For handling class imbalance (SMOTE).
* **matplotlib and seaborn:** For data visualization (confusion matrix).
* **joblib:** For saving and loading Python objects (the trained model).
* **flask:** For creating the web application.

## Potential Improvements

* **More Advanced Feature Engineering:** Incorporate more sophisticated features like the presence of specific spam keywords, unusual punctuation patterns, or the ratio of uppercase letters.
* **Experiment with Different Models:** Try other classification algorithms like Naive Bayes, Support Vector Machines, or ensemble methods.
* **User Interface Enhancements:** Improve the styling and user experience of the web application.
* **Error Handling:** Implement more robust error handling for invalid inputs or model loading issues.
* **Deployment:** Deploy the application to a production server for wider accessibility.

## License

[Your License (e.g., MIT License)](LICENSE) *(Optional: Add a LICENSE file to your repository if you want to specify the license under which your project is distributed.)*

## Author

Abhay Gupta
