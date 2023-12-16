from flask import Flask, render_template
from datanalyzer import DataAnalyzer
from test3 import get_model_accuracy

app = Flask(__name__)

@app.route('/')
def index():
    # Calculate accuracy before bias mitigation
    accuracy_before = get_model_accuracy("DS01.csv", "IMDB_Rating")

    # Perform implicit bias analysis
    data_analyzer = DataAnalyzer(data_path="DS01.csv", target_column="IMDB_Rating")
    data_analyzer.load_data()
    data_analyzer.preprocessing()
    data_analyzer.visualize_data()
    data_analyzer.bias_mitigation()
    accuracy_after = data_analyzer.evaluate_resampled_model()

    # Pass the results to the HTML template
    return render_template('proj.html', accuracy_before=accuracy_before, accuracy_after=accuracy_after)

if __name__ == '__main__':
    app.run(debug=True)
