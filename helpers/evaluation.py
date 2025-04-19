import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
)
from .setup import device, num_labels


# ---------------------- Stock-specific Evaluation functions ---------------------- #


def evaluate_model_by_stock(model, test_loader, test_df, stock_column, post_column, hierarchical=True):
    """
    Robust evaluation function that analyzes model performance by stock

    Args:
        model: The trained HBERT model
        test_loader: DataLoader for the test dataset
        test_df: Original test dataframe with stock and text data
        stock_column: Column name in test_df containing stock symbols
        post_column: Column name in test_df containing post text

    Returns:
        Dictionary with performance metrics by stock
    """
    model.eval()
    all_preds = []
    all_probs = []

    # get all predictions
    with torch.no_grad():
        for batch in test_loader:
            input_ids_list, attention_mask_list, time_values, stock_indices, _ = batch

            # move tensors to device
            input_ids = [ids.to(device) for ids in input_ids_list]
            attention_masks = [mask.to(device) for mask in attention_mask_list]
            time_values = time_values.to(device)
            stock_indices = stock_indices.to(device)
            if not hierarchical:
                input_ids = input_ids[0]
                attention_masks = attention_masks[0]

            # get model outputs
            outputs = model(input_ids, attention_masks, time_values, stock_indices)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # store predictions and probabilities
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # create a results dataframe
    results_df = test_df.copy()
    results_df['prediction'] = all_preds
    results_df['correct'] = results_df['prediction'] == results_df['bert_label']

    # add probability columns
    if len(all_probs) > 0 and len(all_probs[0]) >= 2:
        results_df['prob_decrease'] = [prob[0] for prob in all_probs]
        results_df['prob_increase'] = [prob[1] for prob in all_probs]
        results_df['confidence'] = results_df.apply(
            lambda row: row['prob_decrease'] if row['prediction'] == 0 else row['prob_increase'],
            axis=1
        )

    # group by stock and calculate metrics
    stock_metrics = {}

    for stock, group in results_df.groupby(stock_column):
        # skip stocks with too few samples
        if len(group) < 5:
            print(f"Skipping {stock} - insufficient test samples ({len(group)})")
            continue

        true_labels = group['bert_label'].values
        preds = group['prediction'].values

        # calculate metrics
        try:
            accuracy = accuracy_score(true_labels, preds)
            precision = precision_score(true_labels, preds, average='weighted')
            recall = recall_score(true_labels, preds, average='weighted')
            f1 = f1_score(true_labels, preds, average='weighted')
            cm = confusion_matrix(true_labels, preds)
            report = classification_report(
                true_labels, preds,
                target_names=['Decrease', 'Increase'],
                output_dict=True
            )

            # store metrics
            stock_metrics[stock] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'classification_report': report,
                'samples': len(true_labels)
            }
        except Exception as e:
            print(f"Error calculating metrics for {stock}: {e}")
            continue

    # calculate overall metrics
    overall_true = results_df['bert_label'].values
    overall_preds = results_df['prediction'].values

    overall_metrics = {
        'accuracy': accuracy_score(overall_true, overall_preds),
        'precision': precision_score(overall_true, overall_preds, average='weighted'),
        'recall': recall_score(overall_true, overall_preds, average='weighted'),
        'f1_score': f1_score(overall_true, overall_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(overall_true, overall_preds),
        'classification_report': classification_report(
            overall_true, overall_preds,
            target_names=['Decrease', 'Increase'],
            output_dict=True
        )
    }

    # create a dataframe with metrics for each stock
    metrics_df = pd.DataFrame({
        'Stock': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'Samples': []
    })

    for stock, metrics in stock_metrics.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Stock': [stock],
            'Accuracy': [metrics['accuracy']],
            'Precision': [metrics['precision']],
            'Recall': [metrics['recall']],
            'F1 Score': [metrics['f1_score']],
            'Samples': [metrics['samples']]
        })], ignore_index=True)

    # sort by F1 score for ranking
    metrics_df = metrics_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)

    # print overall metrics
    print("\n=== Overall Performance ===")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"F1 Score: {overall_metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(overall_metrics['confusion_matrix'])

    # print stock ranking
    print("\n=== Stock Performance Ranking (by F1 Score) ===")
    print(metrics_df.to_string(index=False))

    # create bar plot for F1 scores
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Stock', y='F1 Score', data=metrics_df)
    plt.title('F1 Score by Stock')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # create confusion matrices for top and bottom performers
    top_stock = metrics_df.iloc[0]['Stock']
    bottom_stock = metrics_df.iloc[-1]['Stock']

    # top performer confusion matrix
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    top_cm = stock_metrics[top_stock]['confusion_matrix']
    sns.heatmap(
        top_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Decrease', 'Increase'],
        yticklabels=['Decrease', 'Increase']
    )
    plt.title(f'Confusion Matrix - {top_stock} (Best)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # bottom performer confusion matrix
    plt.subplot(1, 2, 2)
    bottom_cm = stock_metrics[bottom_stock]['confusion_matrix']
    sns.heatmap(
        bottom_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Decrease', 'Increase'],
        yticklabels=['Decrease', 'Increase']
    )
    plt.title(f'Confusion Matrix - {bottom_stock} (Worst)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # find and display error cases
    error_cases = results_df[~results_df['correct']].sort_values('confidence', ascending=False)
    print("\n=== High Confidence Error Cases ===")
    if len(error_cases) > 0:
        for _, row in error_cases.head(5).iterrows():
            print(f"\nStock: {row[stock_column]}")
            text = str(row[post_column])
            print(f"Text: {text[:150]}..." if len(text) > 150 else f"Text: {text}")
            print(f"True Label: {'Increase' if row['bert_label'] == 1 else 'Decrease'}")
            print(f"Predicted: {'Increase' if row['prediction'] == 1 else 'Decrease'} (Confidence: {row['confidence']:.2f})")
            print("-" * 50)
    else:
        print("No error cases found!")

    # return results for further analysis
    return {
        'metrics_by_stock': stock_metrics,
        'overall_metrics': overall_metrics,
        'stock_ranking': metrics_df,
        'results_df': results_df
    }


def analyze_stock_specific_errors(results_df, stock_column, post_column):
    """
    Analyze patterns in errors for specific stocks

    Args:
        results_df: DataFrame with prediction results
        stock_column: Column name for stock ticker
        post_column: Column name for post text
    """
    from collections import Counter
    import re

    # calculate error rate by stock
    error_rates = results_df.groupby(stock_column).agg({
        'correct': [lambda x: (~x).mean(), 'count']
    }).reset_index()
    error_rates.columns = [stock_column, 'error_rate', 'samples']

    # sort by error rate
    error_rates = error_rates.sort_values('error_rate', ascending=False)

    # plot error rates
    plt.figure(figsize=(12, 6))
    plt.bar(error_rates[stock_column], error_rates['error_rate'])
    plt.title('Error Rate by Stock')
    plt.ylabel('Error Rate')
    plt.xlabel('Stock')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # analyze error cases for top 3 highest error rate stocks
    top_error_stocks = error_rates.head(3)[stock_column].tolist()

    print("\n=== Stock-Specific Error Analysis ===")
    for stock_name in top_error_stocks:
        stock_errors = results_df[(results_df[stock_column] == stock_name) & (~results_df['correct'])]
        print(f"\nStock: {stock_name} - Error Rate: {error_rates[error_rates[stock_column] == stock_name]['error_rate'].values[0]:.2f}")
        print(f"Number of errors: {len(stock_errors)} out of {error_rates[error_rates[stock_column] == stock_name]['samples'].values[0]} samples")

        # find common words or phrases in error cases
        all_error_text = " ".join(stock_errors[post_column].astype(str).tolist())
        # simple processing to find potential patterns
        words = re.findall(r'\b\w+\b', all_error_text.lower())
        word_counts = Counter(words)

        # print common words (excluding common stopwords)
        stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of', 'this', 'that'
        ])
        relevant_words = [(word, count) for word, count in word_counts.most_common(10) if word not in stopwords]

        print("Common words in error cases:")
        for word, count in relevant_words:
            print(f"  - '{word}': {count} occurrences")

        # print a couple of example error cases
        if len(stock_errors) > 0:
            print("\nExample error cases:")
            for _, row in stock_errors.head(2).iterrows():
                text = str(row[post_column])
                text_preview = text[:150] + "..." if len(text) > 150 else text
                print(f"  - True: {'Increase' if row['bert_label'] == 1 else 'Decrease'}, Predicted: {'Increase' if row['prediction'] == 1 else 'Decrease'}")
                print(f"    Text: {text_preview}")
                print()
    return error_rates


def print_evaluation_report(val_loader, Model, model_dir=None, model=None):
    all_preds = []
    all_labels = []
    if not model:
        model = Model.from_pretrained(model_dir, num_labels=num_labels)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['decrease', 'increase']))

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")


# ---------------------- Time-series Performance Analysis functions ---------------------- #


def analyze_performance_by_time(results_df, date_column):
    """
    Analyze model performance over time.

    Args:
        results_df: DataFrame with prediction results
        date_column: Column name with date information
    """
    # make a copy to avoid modifying the original dataframe
    df = results_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])

    # create day and week strings that can be plotted more easily
    df['day_str'] = df[date_column]
    df['week_str'] = df[date_column].dt.to_period('W').astype(str)

    # group by day
    daily_perf = df.groupby('day_str').agg({'correct': ['mean', 'count']}).reset_index()
    daily_perf.columns = ['day', 'accuracy', 'samples']

    # group by week
    weekly_perf = df.groupby('week_str').agg({'correct': ['mean', 'count']}).reset_index()
    weekly_perf.columns = ['week', 'accuracy', 'samples']

    # sort by chronological order
    daily_perf = daily_perf.sort_values('day')
    weekly_perf = weekly_perf.sort_values('week')

    # plot daily performance
    plt.figure(figsize=(14, 6))
    ax = sns.lineplot(x='day', y='accuracy', data=daily_perf, marker='o')
    plt.title('Model Accuracy by Day')
    plt.ylabel('Accuracy')
    plt.xlabel('day')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # add sample count as annotations
    for i, row in daily_perf.iterrows():
        ax.annotate(
            f"n={row['samples']}",
            (i, row['accuracy']),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )
    plt.show()

    # plot weekly performance
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='week', y='accuracy', data=weekly_perf)
    plt.title('Model Accuracy by Week')
    plt.ylabel('Accuracy')
    plt.xlabel('week')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    # add sample count as annotations
    for i, row in weekly_perf.iterrows():
        ax.annotate(
            f"n={row['samples']}",
            (i, row['accuracy']),
            textcoords="offset points",
            xytext=(0,1),
            ha='center'
        )
    plt.tight_layout()
    plt.show()

    # identify worst performing time periods
    worst_days = daily_perf.sort_values('accuracy').head(3)
    print("\n=== Worst Performing days ===")
    for _, row in worst_days.iterrows():
        print(f"day: {row['day']}, Accuracy: {row['accuracy']:.2f}, Samples: {row['samples']}")

    return {
        'daily_performance': daily_perf,
        'weekly_performance': weekly_perf
    }
