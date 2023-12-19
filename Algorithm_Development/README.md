# Responsive AI - Algorithm Development

## Access the Website here: https://responsiveai-djqrqlwyzzm6bvvcaypdel.streamlit.app/

## Table of Contents
- [About](#about)
- [Navigation](#navigation)
- [Assets](#assets)
- [Data](#data)
- [Notebooks](#notebooks)
- [Header Section](#header-section)
- [Content Section](#content-section)
- [Simulation](#simulation)
- [Metrics](#metrics)
- [Contact Us](#contact-us)
- [Explanation Video](#explanation-video)
- [How to Use](#how-to-use)
- [Contributing](#contributing)

## About
Responsive AI is a project focused on algorithm development for analyzing YouTube comments to provide quality video recommendations to users. The project involves a dedicated Algorithm Development Team and aims to address challenges in content discovery on online video platforms.

## Key Features
- YouTube API integration
- Sentiment analysis of user comments
- Prediction of valuable content
- Bias mitigation strategies

## Navigation
The navigation bar provides easy access to different sections of the project, including Home, About Us, Project details, Simulation, Metrics, and Contact Us.

## Assets
The Assets section includes Lottie animations and a header section with project information.

## Data
The data used in the project is stored in the data folder. The `analysis_details.csv` file contains information about all the videos that have been analyzed.

## Notebooks
The notebooks folder contains Python files for different parts of the analysis.
- `Simple_Bias_detection.ipynb`: This notebook explores basic bias detection techniques.
- `Sentiment_Analysis_Youtube.ipynb`: This notebook performs sentiment analysis on YouTube comments.
- `Toxity_Measure.ipynb`: This notebook is an additional file that measures the toxicity of comments and makes the developers delete toxic comments for their video.

## Header Section
The header section contains a welcome message and essential details about the project, including mentors and industry experts.

## Content Section
The content section provides detailed information about the project, including an abstract, introduction, solution details, bias dimensions, sentiment analysis, and latent bias.

## Simulation
The Simulation section allows users to input a YouTube API key and simulate video queries to display sentiment analysis results. After the Simulation, a download button will appear. It will allow us to download the data in csv format. Store the file in the same directory (where your `app.py` is present) under the data folder.

## Metrics
The Metrics section provides insights into the sentiment analysis results, including total comments analyzed, average sentiment scores, and visualizations such as bar charts, box plots, histograms, and pie charts.

## Contact Us
The Contact Us section provides email addresses for general inquiries, technical support, and project-related queries.

## Explanation Video
[<img src="https://github.com/RincisM/Responsive_AI/blob/main/Algorithm_Development/images/Algorithm_Development_Title_Page.png?raw=true" width="600" height="300"/>](https://www.youtube.com/embed/l8qB0fx2SOc)

## How to Use
1. Clone the repository: `git clone https://github.com/RincisM/Responsive_AI.git`
2. Navigate to Algorithm_Development directory.
3. Install required dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app.py`

## Contributing
If you'd like to contribute to this project, please follow these guidelines:
1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Create a pull request
