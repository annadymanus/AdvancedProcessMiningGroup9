import os

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from skmultilearn.model_selection import iterative_train_test_split

np.random.seed(1868)

thresholds = list(reversed([0.7 + x * 0.01 for x in range(0, 29)] + [0.98 + x * 0.001 for x in range(0, 20)]))


def parse_filename(filename):
    _, company, tweets, direction, _ = filename.split('.')[0].split('-')
    return company, tweets, direction


def cross_validate_nli_template(df, test_df, df_nli_template):
    results = []
    test_results = []
    for header in list(df_nli_template):
        header_results = []
        hypotheses = df_nli_template[header].dropna().values.tolist()
        if len(hypotheses) == 0:
            continue
        for hypothesis in hypotheses:
            num_positive_instances = len(df[df[header] == 1].index)
            if num_positive_instances >= 3:
                y_true = df[header].astype('int64')
                y_probabilities = df['pred_' + header + '_' + hypothesis].astype('float64')
                skf = StratifiedKFold(n_splits=min(num_positive_instances, 5), shuffle=True)
                threshold_results = []
                for threshold in thresholds:
                    fold_results = []
                    for i, splits in enumerate(skf.split(y_probabilities, y_true)):
                        _, test = splits
                        y_predicted = y_probabilities[test].apply(lambda x: 1 if x > threshold else 0)
                        fold_results.append((threshold,) + evaluate_predictions(y_true[test], y_predicted))
                    threshold_results.append(tuple(map(lambda y: sum(y) / float(len(y)), zip(*fold_results))))
                optimal_threshold = max(threshold_results, key=lambda item: item[1])[0]
                predictions = df['pred_' + header + '_' + hypothesis].apply(lambda x: 1 if x > optimal_threshold else 0)
                evaluation = evaluate_predictions(df[header].astype('int64'), predictions.astype('float64'))
                results.append((header, hypothesis, optimal_threshold) + evaluation)
                header_results.append((header, hypothesis, optimal_threshold) + evaluation)
        
        #Run on test_df with best hypothesis and threashold
        if len(header_results) > 0:
            best_combination = max(header_results, key=lambda item: item[3])
            best_hypothesis = best_combination[1]
            best_threshold = best_combination[2]
            y_true = test_df[header].astype('int64')
            y_probabilities = test_df['pred_' + header + '_' + best_hypothesis].astype('float64')
            y_predicted = y_probabilities.apply(lambda x: 1 if x > best_threshold else 0)
            evaluation = evaluate_predictions(y_true, y_predicted)
            test_results.append((header, best_hypothesis, best_threshold) + evaluation)

    df_results = pd.DataFrame(results, columns=['Header', 'Label', 'Optimal Threshold', 'MCC', 'Accuracy',
                                                'Balanced Accuracy', 'F1', 'Items'])
    test_df_results = pd.DataFrame(test_results, columns=['Header', 'Label', 'Optimal Threshold', 'MCC', 'Accuracy',
                                                'Balanced Accuracy', 'F1', 'Items'])
    return df_results, test_df_results


def evaluate_predictions(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    items = len(y_true[y_true == 1].index)
    return mcc, accuracy, balanced_accuracy, f1, items


if __name__ == '__main__':

    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists(os.path.join('results', 'nli-cv')):
        os.mkdir(os.path.join('results', 'nli-cv'))

    for filename in os.listdir(os.path.join('data', 'predicted', 'combined', 'train')):
        company, tweets, direction = parse_filename(filename)
        df = pd.read_excel(os.path.join('data', 'predicted', 'combined', 'train', filename), index_col=[0]).reset_index(drop=True)
        test_df = pd.read_excel(os.path.join('data', 'predicted', 'combined', 'test', filename), index_col=[0]).reset_index(drop=True)
        
        # merge df and test_df and split again with stratified train-test split
        df_combined = pd.concat([df, test_df], ignore_index=True)
        company_index = list(df_combined.columns).index('company')
        first_pred_index = 0
        for i, column in enumerate(df_combined.columns):
            if column.startswith('pred_'):
                first_pred_index = i
                break
        class_names = list(df_combined.columns)[company_index+1:first_pred_index]
        print(class_names)
        #Drop all rows of classes with less than 3 positive instances
        for class_name in class_names:
            if len(df_combined[df_combined[class_name] == 1].index) < 3:
                df_combined = df_combined[df_combined[class_name] == 0]
                print(f'Dropped {class_name}')
        df_combined = df_combined.reset_index(drop=True)
        df_values, _, test_df_values, _ = iterative_train_test_split(df_combined.values, df_combined[class_names].values, test_size=0.33)   
        df = pd.DataFrame(df_values, columns=df_combined.columns)
        test_df = pd.DataFrame(test_df_values, columns=df_combined.columns)
        
        nli_template = 'twcs-{}-nli.xlsx'.format(company)
        df_nli_template = pd.read_excel(os.path.join('data', 'nli-templates', direction, nli_template))
        df_results, test_df_results = cross_validate_nli_template(df, test_df, df_nli_template)
        test_df_results.to_excel(os.path.join('results', 'nli-cv', 'combined', 'test', 'test_cv_results-{}-{}.xlsx'.format(company, direction)), index=False)
        outfile = 'train_cv_results-{}-{}.xlsx'.format(company, direction)
        writer = pd.ExcelWriter(os.path.join('results', 'nli-cv', 'combined', 'train', outfile), engine='xlsxwriter')
        df_results.to_excel(writer, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        for column in ['E', 'F', 'G', 'H']:
            worksheet.conditional_format('{}2:{}{}'.format(column, column, str(len(df_results.index) + 1)),
                                         {'type': '3_color_scale'})
        writer.close()
