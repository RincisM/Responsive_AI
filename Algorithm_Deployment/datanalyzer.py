import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

class DataAnalyzer:
    def __init__(self, data_path, target_column='IMDB_Rating', random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state
        self.data = None
        self.df = None
        self.le = LabelEncoder()
        self.model = None
        self.features = None
        self.target = None
        self.train_data = None
        self.test_data = None
        self.train_target = None
        self.test_target = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.df = pd.DataFrame(self.data)
        print(self.data)

    def preprocessing(self):
        # Print the head of the dataset before preprocessing
        print("Dataset head before preprocessing:")
        print(self.df.head())

        # Assume '?' represents missing values and replace with NaN
        self.df.replace('?', pd.NA, inplace=True)

        # Handling missing values
        self.df.dropna(inplace=True)

        # Encoding categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.le.fit_transform(self.df[col])

        self.features = self.df.drop(columns=[self.target_column])
        self.target = self.le.fit_transform(self.df[self.target_column])

        # Print the head of the dataset after preprocessing
        print("\nDataset head after preprocessing:")
        print(self.df.head())
        print(self.df.columns)

    def visualize_data(self):
        # Visualization of data for each column against the target variable using histograms
        num_columns = len(self.features.columns)
        num_rows = (num_columns + 1) // 2
        plt.figure(figsize=(15, 5 * num_rows))

        for i, col in enumerate(self.features.columns, start=1):
            plt.subplot(num_rows, 2, i)
            for label in self.df[self.target_column].unique():
                sns.histplot(self.df[self.df[self.target_column] == label][col], bins=20, label=f'{self.target_column}={label}', kde=False)
            plt.title(f'Histogram: {col} vs. {self.target_column}')
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def bias_mitigation(self):
        # Assuming multi-class classification for bias mitigation
        class_counts = self.df[self.target_column].value_counts()

        # Check if there are enough instances for each class
        if any(count == 0 for count in class_counts):
            print("Insufficient instances for one or more classes. Bias mitigation cannot be performed.")
            return

        # Resample the majority class for balancing
        max_class_count = max(class_counts)
        df_resampled = pd.DataFrame()

        for label, count in class_counts.items():
            if count < max_class_count:
                label_data = self.df[self.df[self.target_column] == label]
                resampled_data = resample(label_data, replace=True, n_samples=max_class_count, random_state=self.random_state)
                df_resampled = pd.concat([df_resampled, resampled_data])

        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(
            df_resampled.drop(columns=[self.target_column]),
            self.le.fit_transform(df_resampled[self.target_column]),
            test_size=0.2, random_state=self.random_state
        )

        self.model = RandomForestClassifier(random_state=self.random_state)
        try:
            self.model.fit(self.train_data, self.train_target)
            print("Model trained successfully.")
        except Exception as e:
            print(f"Error during model training: {e}")
            return

    def evaluate_resampled_model(self):
        if self.model is None:
            print("Model has not been trained. Please run bias_mitigation first.")
            return None

        predictions_resampled = self.model.predict(self.test_data)
        accuracy_resampled = accuracy_score(self.test_target, predictions_resampled)
        report_resampled = classification_report(self.test_target, predictions_resampled)

        print(f'Resampled Model Accuracy: {accuracy_resampled}\nResampled Classification Report:\n{report_resampled}')

        return accuracy_resampled  # Return accuracy

    def check_implicit_bias(self):
        if self.target_column not in self.df.columns:
            print(f"Target column '{self.target_column}' not found in the dataset.")
            return

        for col in self.features.columns:
            if col != self.target_column:
                print(f"\nAnalyzing Implicit Bias for '{col}' vs. '{self.target_column}':")
                plt.figure(figsize=(12, 6))
                for label in self.df[self.target_column].unique():
                    sns.histplot(self.df[self.df[self.target_column] == label][col], bins=20, label=f'{self.target_column}={label}', kde=False)
                plt.title(f'Histogram: {col} vs. {self.target_column}')
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.legend()
                plt.show()

                # Check for statistical parity difference
                for label in self.df[self.target_column].unique():
                    label_data = self.df[self.df[self.target_column] == label][col]
                    overall_data = self.df[col]

                    favorable_outcome_rate_label = label_data.mean()
                    favorable_outcome_rate_overall = overall_data.mean()

                    statistical_parity_difference = favorable_outcome_rate_label - favorable_outcome_rate_overall

                    print(f'Statistical Parity Difference for {col} and {self.target_column}={label}: {statistical_parity_difference}')

                    # Check for bias
                    if abs(statistical_parity_difference) <= 0.1:
                        print(f'NO SIGNIFICANT BIAS IS DETECTED IN {col} COLUMN for {self.target_column}={label}.')
                    else:
                        print(f'IMPLICIT BIAS IS DETECTED IN {col} COLUMN for {self.target_column}={label}.')

# Example usage:
#data_analyzer = DataAnalyzer(data_path="Employee.csv", target_column="LeaveOrNot")
#data_analyzer.load_data()
#data_analyzer.preprocessing()
#data_analyzer.visualize_data()
#data_analyzer.bias_mitigation()
#accuracy = data_analyzer.evaluate_resampled_model()
#print(f"Accuracy: {accuracy}")
##data_analyzer.check_implicit_bias()

