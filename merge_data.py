
#Merge csv files in labeled directory with those in labeled/new_labels directory

import os
import csv
import pandas as pd
from collections import defaultdict

labeled_path = os.path.join(os.getcwd(), 'data','labeled')
new_labeled_path = os.path.join(os.getcwd(), 'data', 'labeled', 'new_labels')

def get_data(labeled_path):
    companies = ["AmazonHelp", "AppleSupport", "SpotifyCares"]
    labeled_data = defaultdict(dict)
    for file in os.listdir(labeled_path):
        if file.endswith('.xlsx'):
            for company in companies:
                if company in file:
                    print(os.path.join(labeled_path, file))
                    if "outbound" in file:
                        labeled_data[company]["outbound"] = pd.read_excel(os.path.join(labeled_path, file))
                    elif "inbound" in file:
                        labeled_data[company]["inbound"] = pd.read_excel(os.path.join(labeled_path, file))
    return labeled_data

def merge_data(labeled_data, new_data):
    for company in labeled_data:
        for direction in labeled_data[company]:
            labeled_data[company][direction] = pd.concat([labeled_data[company][direction], new_data[company][direction]])
    return labeled_data

def train_test_split_data(merged_data):
    train_data = defaultdict(dict)
    test_data = defaultdict(dict)
    for company in merged_data:
        for direction in merged_data[company]:
            train_data[company][direction] = merged_data[company][direction].sample(frac=0.66)
            test_data[company][direction] = merged_data[company][direction].drop(train_data[company][direction].index)
    return train_data, test_data

def save_data(data, save_path):
    for company in data:
        for direction in data[company]:
            data[company][direction].to_excel(os.path.join(save_path, "twcs-" + company + "-n-" + direction + ".xlsx"), index=False)


data = get_data(labeled_path)
new_data = get_data(new_labeled_path)
merged_data = merge_data(data, new_data)
train_data, test_data = train_test_split_data(merged_data)
save_data(train_data, os.path.join(labeled_path, "combined","train"))
save_data(test_data, os.path.join(labeled_path, "combined","test")) 

