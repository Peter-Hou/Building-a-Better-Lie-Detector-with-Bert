
"""Read Text Files for BERT Processing

This script reads text files from a given directory structure, separating them into
'deceptive' and 'truthful' categories based on their filenames. Files starting with 'd'
are considered deceptive, and those starting with 't' are considered truthful. The texts
are then saved into two pickle files for further processing with BERT.

The expected directory structure is:
- base_directory/
  - negative_polarity/
    - deceptive_folder/
    - truthful_folder/
        - fold1/
        - fold2/
        - fold3/
        - fold4/
        - fold5/
  - positive_polarity/
    - deceptive_folder/
    - truthful_folder/
        - fold1/
        - fold2/
        - fold3/
        - fold4/
        - fold5/

Each fold contains multiple '.txt' files, each with a single line of text.
"""

import os
import pickle

def read_texts_from_folders(base_path, starting_letter):
    """
    Reads all text files that start with a specific letter from a given base path.
    
    Args:
        base_path (str): The base directory containing text files.
        starting_letter (str): The starting letter of files to read ('d' for deceptive, 't' for truthful).
        
    Returns:
        list of str: A list of strings, each representing the content of a text file.
    """

    texts = []
    # Walk through the directory
    for root, dirs, files in os.walk(base_path):
        for file_name in files:
            # Check if the file name starts with the given letter
            if file_name.startswith(starting_letter) and file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                # Open and read the file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    texts.append(file.read().strip())
    return texts

def write_to_pickle(file_path, data):
    """
    Writes the given data to a pickle file.
    
    Args:
        file_path (str): The path to the pickle file to be written.
        data (Any): The data to write to the pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    file_save_path = "/Users/peterhou/DataScienceLife/Building-a-Better-Lie-Detector-with-Bert/data/"
    dec_data = read_texts_from_folders(file_save_path + "op_spam_v1.4", "d")
    tru_data = read_texts_from_folders(file_save_path + "op_spam_v1.4", "t")
    write_to_pickle(file_save_path + "berted_deception.pkl", dec_data)
    write_to_pickle(file_save_path + "berted_truthful.pkl", tru_data)

