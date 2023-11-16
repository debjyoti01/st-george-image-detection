## St. George Image Detection

### Problem Statement
The task involves detecting the presence of St. George in images.

### Inputs
Two CSV files, `georges.csv` and `non-georges.csv`, contain lists of images with and without St. George, respectively.

### Output
Predict George is present in image or not.

### Language and Editor Used
 - **Python 3**
 - **Jupyter Notebook**

### Approach
The solution comprises two Python files:
1. **process_the_csv_file.ipynb**: this python code Processes the given CSV to a more easy to accessible csv file.
2. **apply_classification_algorithm.ipynb**: Applies multiple classification algorithms to determine the best performer for this dataset.

### File structure
![image](https://github.com/debjyoti01/st-george-image-detection/assets/120040021/ac3735eb-66fb-43cd-979c-4bd41fe79485)

All script are done inside 'Data Engineer Test' Folder

### `process_the_csv_file.ipynb`
### Objective
Convert image URLs in the CSV to local paths by extracting images from a provided zip file.
### Prerequisites

- **Python 3.x**
- **Pandas Library**

To install Pandas, use the following command:

```bash
pip install pandas
```

#### Running the Code
1. **Install Required Libraries**: Use `pip` to install the libraries mentioned in the prerequisites.
2. **Run the Notebook**: Execute the cells in `process_the_csv_file.ipynb`.
3. **Customization**: Modify the script in the notebook to suit your dataset paths, if necessary.

#### How the Script Works

1. The function `process_csv` takes 3 parameters: `csv_file_path`, `local_path`, and `output_csv_file`, whose values are described above.

2. It then stores the column name that will be added.

3. The script reads the CSV file into a dataframe and adds the specified column name.

4. The `extract_filename()` function extracts the base file name and concatenates it with `local_path`.

5. Finally, the modified dataframe is stored in the file specified by `output_csv_file`.


### `apply_classification_algorithm.ipynb`

### Objective
This Python script aims to classify images as containing St. George or not using machine learning algorithms. It involves extracting color histogram features and applying various classifiers for prediction, aiming to determine the presence of St. George in the provided dataset.

### Prerequisites
- **Python 3.x**: Install Python from [python.org](https://www.python.org/downloads/).
- **Required Libraries**:
    - Install the necessary libraries using `pip`:
        ```bash
        pip install opencv-python numpy pandas scikit-learn matplotlib
        ```
    - Ensure you have the following libraries:
        - `cv2` (OpenCV)
        - `numpy`
        - `pandas`
        - `scikit-learn` (sklearn)
        - `matplotlib`

### Usage
1. **Load Data**: Access the CSV files `updated_georges.csv` and `updated_non_georges.csv`, containing image paths for St. George and non-St. George images.
2. **Feature Extraction**: Extract color histogram features from each image.
3. **Classification**:
    - Utilize multiple classifiers such as K-Nearest Neighbors, Support Vector Machine, Random Forest, Decision Tree, Naive Bayes, and Logistic Regression.
    - Evaluate model performance using accuracy scores and classification reports.
4. **Example Usage**:
    - Employ the `classify_image_rf` function to classify new images.
    - Example: `classify_image_rf("path_to_image.jpg", rf_model)`.

### Files Included
- `apply_classification_algorithm.ipynb`: Python script containing the image classification code.
- `updated_georges.csv`, `updated_non_georges.csv`: CSV files containing image paths.
- `george_test_task/`: Directory holding test images.

### Running the Code
1. **Install Required Libraries**: Use `pip` to install the libraries mentioned in the prerequisites.
2. **Run the Notebook**: Execute the cells in `apply_classification_algorithm.ipynb`.
3. **Customization**: Modify the script in the notebook to suit your dataset paths, if necessary.

### Steps in Detail

1. Import the libraries
    - Import necessary libraries.
  
2. Get the data in data frame
    - Collect data using the `process_the_csv_file.ipynb` script and store it in a dataframe.

3. Labeling the data
    - Assign label 1 for George images and label 0 for non-George images.

4. Concatenate and shuffle data frames
    - Merge the two data frames into a single `data` variable and shuffle them.

5. Check for NULL data
    - Verify if there are any NULL or missing values in the dataset.

6. Split the data
    - Divide the dataset into an 80:20 ratio for training and testing.

7. Define `extract_color_histogram()` function
    - Create a function to extract the color histogram of images.
    - Arguments: image path and a default argument named 'bin'.

8. Extract features using `extract_color_histogram()`
    - Apply the `extract_color_histogram()` function to obtain features for all images.

9. Apply different classification algorithms
    - Utilize various classifiers and display their accuracy scores and classification reports.

10. Select the best classifier
    - Choose the Random Forest classifier based on its highest accuracy.

11. Predict using the Random Forest model
    - Use the Random Forest model to predict the label of a new picture.

12. Compute the ROC curve
    - Calculate the Receiver Operating Characteristic (ROC) curve for the Random Forest predicted model.


### Performance Summary:
- **Random Forest Classifier** has the highest accuracy (77.5%) and decent precision, recall, and F1-scores for both classes. It outperforms most other classifiers.
- **Support Vector Machine (SVM) Classifier** demonstrates strong accuracy (71.7%) with a good balance between precision and recall.
- **Logistic Regression** and **KNN Classifier** show comparable performance, both achieving around 69-70% accuracy.
- **Decision Tree Classifier** and **Naive Bayes Classifier** have relatively lower accuracies compared to other models.

### Conclusion:
1. **Best Performing Model:** The Random Forest Classifier stands out with the highest accuracy among the tested classifiers, offering a good balance between precision and recall for both classes.
2. **Feature Engineering:** Utilizing color histograms proved effective in capturing image features for the models. Further feature engineering, like extracting texture or shape features, could potentially enhance model performance.
3. **Model Complexity:** More complex models like Random Forest and SVM proved more effective in capturing the underlying patterns in the image features.
4. **Class Imbalance:** The difference in class distribution might affect classifier performance, especially for the minority class (presence of St. George), as seen in Naive Bayes, which struggled with recall for the minority class.
5. **Future Improvements:** Consider exploring other image feature extraction techniques or advanced machine learning models, or even neural networks, to further enhance classification performance. Handling class imbalance via sampling techniques or adjusting class weights might also improve the performance of models on the minority class.

### Overall:
The Random Forest Classifier seems to be the most suitable for this task based on the provided data and features. However, further experimentation with feature engineering and model tuning could potentially yield even better results. The choice of the model ultimately depends on the balance between precision, recall, and computational complexity for the specific use case or application.


