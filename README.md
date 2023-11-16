## St. George Image Detection

### Problem Statement
The task involves detecting the presence of St. George in images using Python and Jupyter Notebook. Two CSV files, `georges.csv` and `non-georges.csv`, contain lists of images with and without St. George, respectively.

### Approach
The solution comprises two Python files:
1. **process_the_csv_file.ipynb**: Processes the given CSV to a more easy to accessible format.
2. **apply_classification_algorithm.ipynb**: Applies multiple classification algorithms to determine the best performer for this dataset.

### `process_the_csv_file.ipynb`
- **Objective**: Convert image URLs in the CSV to local paths by extracting images from a provided zip file.
- **Prerequisites**:
    - Python 3.x
    - Pandas Library

    To install Pandas, use:
    ```
    pip install pandas
    ```

#### Steps to Follow
1. **Installation**: Install the Pandas library.
2. **Script Customization**:
    - Modify the script based on your requirements:
        - `csv_file_path`: Input CSV file path.
        - `output_csv_file`: Output CSV file path.
        - `local_path`: Path where the images are stored.
        - `column_names`: CSV column name under which the URLs will be written.

#### How the Script Works

1. The function `process_csv` takes 3 parameters: `csv_file_path`, `local_path`, and `output_csv_file`, whose values are described above.

2. It then stores the column name that will be added.

3. The script reads the CSV file into a dataframe and adds the specified column name.

4. The `extract_filename()` function extracts the base file name and concatenates it with `local_path`.

5. Finally, the modified dataframe is stored in the file specified by `output_csv_file`.

