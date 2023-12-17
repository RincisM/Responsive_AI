import requests
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import plotly.express as px


#Coding - Sentiment Analysis
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_youtube_comments(api_key, education_query, max_results=50):
    # Create a YouTube API client
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments_dataset = []
    try:    
        search_response = youtube.search().list(
            q=education_query,
            type="video",
            part=["snippet"],
            maxResults=max_results,
            relevanceLanguage='en'  
        ).execute()
       
        for item in search_response.get('items', []):
            video_id = item['id']['videoId']
            link = f'https://www.youtube.com/watch?v={video_id}'

            # Get comments for the video
            comments_request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText'
            )
            try:
                comments_response = comments_request.execute()            
                for comment in comments_response.get('items', []):
                    comment_text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                    try:
                        comment_language = detect(comment_text)
                        if comment_language == 'en':
                            comments_dataset.append({
                                'video_id': video_id,
                                'link': link,
                                'comment': comment_text,
                            })
                    except:
                        pass  

            except HttpError as e:
                # Handle commentsDisabled error
                if "commentsDisabled" in str(e):
                    st.warning(f"Comments are disabled for the video: {video_id} ({link})")
                else:
                    st.error(f"An error occurred during comments request: {str(e)}")

    except HttpError as e:
        st.error(f"An error occurred during search request: {str(e)}")

    df_comments = pd.DataFrame(comments_dataset)

    return df_comments

def analyze_sentiment(df_comments):
    
    grouped_df = df_comments.groupby('video_id').filter(lambda x: len(x) >= 4)
    analyzer = SentimentIntensityAnalyzer()
    def get_sentiment_label(compound_score):
        if compound_score >= 0.05:
            return 'Positive'
        elif -0.05 < compound_score < 0.05:
            return 'Neutral'
        else:
            return 'Negative'

    grouped_df['sentiment_score'] = grouped_df['comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    grouped_df['sentiment_label'] = grouped_df['sentiment_score'].apply(get_sentiment_label)

    return grouped_df[['video_id', 'comment', 'sentiment_score', 'sentiment_label']]



def calculate_final_sentiment(grouped_df):
    
    grouped_df['positive_score'] = grouped_df['sentiment_score'][grouped_df['sentiment_label'] == 'Positive']
    grouped_df['negative_score'] = grouped_df['sentiment_score'][grouped_df['sentiment_label'] == 'Negative']

    video_sentiment_scores = grouped_df.groupby('video_id')[['positive_score', 'negative_score']].sum().fillna(0)
    video_sentiment_scores['negative_score'] = video_sentiment_scores['negative_score'] * -1

    video_sentiment_scores['final_sentiment'] = \
        video_sentiment_scores.apply(lambda row: 'Positive' if row['positive_score'] > row['negative_score'] else 'Negative', axis=1)

    return video_sentiment_scores[['positive_score', 'negative_score', 'final_sentiment']]



def selection_positive_comments_video(video_sentiment_scores):
    
    video_sentiment_scores = video_sentiment_scores.drop(columns=['negative_score'])
    video_sentiment_scores = video_sentiment_scores[video_sentiment_scores['final_sentiment'] == 'Positive']
    video_sentiment_scores = video_sentiment_scores.sort_values(by='positive_score', ascending=False)    
    video_sentiment_scores['link'] = video_sentiment_scores.index.map(lambda x: f'https://www.youtube.com/watch?v={x}')

    return video_sentiment_scores['link']

def save_analysis_details(video_sentiment_scores, file_path="analysis_details.csv"):
    # Save the analysis details to a CSV file
    video_sentiment_scores.to_csv(file_path, index=True)

def download_file(file_path, file_name="analysis_details.csv"):
    # Provide a download link for the user
    with open(file_path, "rb") as file:
        st.download_button(
            label="Download Analysis Details",
            data=file,
            key="download_button",
            file_name=file_name,
            mime="text/csv",
        )

#print(filtered_videos)


# Set page configuration
st.set_page_config(page_title="Responsive AI", page_icon=":graduation_cap:", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def simulate_and_display_results(api_key, user_query):
    # Fetch comments for the simulated videos
    df_comments = fetch_youtube_comments(api_key=api_key, education_query=user_query)  
    # Perform sentiment analysis for each video
    grouped_df = analyze_sentiment(df_comments)
    video_sentiment_scores = calculate_final_sentiment(grouped_df)
    filtered_videos = selection_positive_comments_video(video_sentiment_scores)
    filtered_videos_df = filtered_videos.reset_index().rename(columns={0: 'link'})
    #total_positive_score = video_sentiment_scores['positive_score'].sum()

    return filtered_videos_df, video_sentiment_scores, grouped_df

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #1E88E5;">Responsive AI - Algorithm Development</h1>
        <br>
    </div>
    """,
    unsafe_allow_html=True,
)

# #sidebar
# nav_options = ["Home", "About Us", "Project", "Simulation", "Contact Us"]
# selected_nav = st.sidebar.radio("Navigation", nav_options)

# Navigation bar
selected_nav = option_menu(
    menu_title=None,
    options = ["Home", "About Us", "Project", "Simulation","Metrics","Contact Us"],
    icons=["house", "person-badge", "book", "pc-display", "clipboard2-pulse", "flag"],
    default_index=0,
    orientation="horizontal",
    # styles={
    #     "container": {"padding": "0!important", "background-color": "#fafafa"},
    #     "icon": {"color": "orange", "font-size": "25px"}, 
    #     "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    #     "nav-link-selected": {"background-color": "green"},
    # }
)

# --Assets--

lottie_coding = "https://lottie.host/37ba51a6-3953-4bf7-a72b-09bd2a22ab3b/yHsfbMKPn1.json"

# --Header Section--
with st.container():
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>Welcome to the Home of Algorithm Development Team</h2>
            <p>Analysis of Youtube Comments to provide Quality Videos to the User</p>
            <p><strong>Mentor:</strong> Prof. Dr. Deivamani M</p>
            <p><strong>Industry Expert:</strong> Mr. Muthumani M</p>
            <p>Visit our Github Page by clicking <a href="https://github.com/RincisM/Responsive_AI/tree/main/Algorithm_Development" target="_blank">here</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

# --Content Section--
if selected_nav == "Home":
    # Home content
    st_lottie(lottie_coding, height=300, key="coding")

elif selected_nav == "About Us":

    st.markdown(
    """
    <div style="text-align: center;">
        <h2><b>About Us</b></h2>
        <p>Here you can find about the Algorithm Development Team</p>
    </div>
    """,
    unsafe_allow_html=True,
    )

    # Styling for the avatars and brief info
    avatar_style ="""
        <style>
            img {
                border-radius: 50%;
                box-shadow: 0px 0px 15px 0px rgba(0, 0, 0, 0.3);
                margin-bottom: 10px;
                display: block;
                margin: 0 auto;
            }
            .avatar-container {
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 100%;
            }
            .avatar-link {
                color: #1E88E5;
                text-decoration: none;
                font-weight: bold;
                display: block;
            }
        </style>
    """
    st.markdown(avatar_style, unsafe_allow_html=True)

    # Dummy avatars and brief info
    col1, col2, col3 = st.columns(3)

    # Define the target URLs for each image
    url_kailash = "https://www.example.com/kailash"
    url_rincis = "https://www.linkedin.com/in/rincis-m"
    url_karthikkeyan = "https://www.linkedin.com/in/karthikkeyan-natarajan-014283228/"

    # Use markdown and HTML to create the hyperlinks with styling
    with col1:
        st.image("./images/avatar.png", use_column_width="auto")
        st.markdown(f"<div class='avatar-container'><a class='avatar-link' href='{url_kailash}'>Kailash Chandran</a><p>kailashchandran99999@gmail.com</p></div>", unsafe_allow_html=True)
        with st.expander("Expand to know more about me"):
            st.write(
                "I would like to Contribute to the growth of a company by applying my technical skills."
                "I have a strong passion for problem solving, and a strong desire to continually learn and adapt to emerging technologies."
            )

    with col2:
        st.image("./images/avatar.png", use_column_width="auto")
        st.markdown(f"<div class='avatar-container'><a class='avatar-link' href='{url_rincis}'>Rincis Melvin M</a><p>rincismelvin25@gmail.com</p></div>", unsafe_allow_html=True)
        with st.expander("Expand to know more about me"):
            st.write(
                "A dedicated and adaptable Master's in Computer Applications (MCA) student."
                "I am enthusiastic about the opportunity to contribute my technical skills and passion for innovation." 
                "With a background in Physics and a strong foundation in programming, I bring a unique perspective to the table and am committed to continual learning and growth within the dynamic field of technology." 
            )

    with col3:
        st.image("./images/avatar.png", use_column_width="auto")
        st.markdown(f"<div class='avatar-container'><a class='avatar-link' href='{url_karthikkeyan}'>Karthikkeyan N T</a><p>ntkarthi29@gmail.com</p></div>", unsafe_allow_html=True)
        with st.expander("Expand to know more about me"):
            st.write(
                "I intent to be a part of an organisation where I can constantly learn and develop my texhnical"
                "and management skills and make best use of it for the growth of the organisation."
                "I look forward to establishing myself by adapting new technologies as well."
            )

    
    st.write("\n\n")

    st.markdown(
    """
    <div style="text-align: center;">
        <p>We, the students of the College of Engineering, Anna University, are currently pursuing Master of Computer Applications.
           Under the guidance of Dr. Deivamani M (Assistant Professor) and Mr. Muthumani M (Industry Expert) in our academic journey, 
           we have collaboratively worked on enhancing Artificial Intelligence, focusing on the responsible identification and mitigation of bias utitilizing it in recommendation systems.</p>
    </div>
    """,
    unsafe_allow_html=True,
    )


elif selected_nav == "Project":
    st.markdown(
    """
    <div style="text-align: center;">
        <h2><b>Project</b></h2>
        <p>Youtube Sentiment Analysis Application</p>
    </div>
    """,
    unsafe_allow_html=True,
    )
    
    # Abstract
    st.subheader("**Abstract**")
    st.write(
    "In the era of digital content consumption, online video platforms such as YouTube have become prominent "
    "sources for information, entertainment, and education. As the volume of content on these platforms continues "
    "to grow, users face the challenge of efficiently discovering videos that align with their interests and preferences. "
    "This project, titled 'Responsive AI,' aims to address this challenge by leveraging the YouTube Data API to analyze "
    "user comments, perform sentiment analysis, and predict videos with the most valuable content.\n\n"
    
    "Our project involves a collaborative effort by a dedicated Algorithm Development Team. We employ Google Console's API key "
    "to access YouTube's vast dataset, allowing us to retrieve and analyze comments associated with each video. The key focus is "
    "on understanding the sentiments expressed in these comments to gauge user satisfaction and engagement.\n\n"
    
    "The core objective is to identify and recommend videos that are deemed to contain the best content and are likely to be "
    "valuable to a wide audience. We utilize sentiment analysis techniques to extract insights from user comments, enabling us "
    "to make informed predictions about the overall quality and usefulness of a video."
    )

    st.subheader("**Introduction**")
    st.subheader("**YouTube's Recommendation System**")
    st.write(
    "YouTube's recommendation system traditionally relies on user engagement data, such as watch history, likes, and clicks." 
    "While effective to a certain extent, this approach may reinforce existing biases and limit the diversity of content exposure."
    )
    st.subheader("**Responsive AI Solution**")
    st.write(
    "The **Responsive AI** project addresses these challenges by introducing sentiment analysis of user comments to enhance video recommendations." 
    "By leveraging natural language processing techniques, the project aims to understand the sentiments expressed in comments associated with each video."
    )
    st.subheader("**Bias and its Dimensions**")
    st.write(
    "Bias refers to the systematic favoritism or prejudice toward certain factors that influence decision-making processes." 
    "In the context of content recommendation, biases can manifest in different dimensions."
    )
    st.write("**Demographic Bias**: Demographic bias occurs when recommendations disproportionately favor or exclude certain user demographics." 
    "This bias may result from a lack of diversity in training data or implicit biases in user interactions."
    )
    st.write("**Sentiment Bias**: Sentiment bias is rooted in the emotional tone of content."
    "It arises when recommendation algorithms favor content with specific sentiment characteristics, potentially leading to a homogenized user experience."
    )
    st.write("**Latent Bias**: Latent bias is less explicit and originates from hidden patterns in data."
    "It can emerge when algorithms inadvertently learn and perpetuate biases present in training data."
    )

    st.subheader("**Understanding Sentiment and Latent Bias**")
    st.write(
    "Sentiment analysis involves assessing the emotional tone of user comments. Positive, Negative, or Neutral sentiments are assigned to comments, enabling"
    "a nuanced understanding of user engagement. Latent bias, on the other hand, requires a deep dive into underlying patterns that may not be immediately evident."
    "Latent bias can significantly impact sentiment analysis models, especially when analyzing user comments. User sentiments are often subjective and can be influenced by societal, cultural, or contextual factors that may not be immediately apparent."
    "Latent bias is challenging to identify because it may not be explicitly represented in the training data. Traditional bias mitigation techniques may not effectively address latent bias, making it essential to adopt specific strategies for sentiment analysis."
    )


    st.divider()

    # C4 diagram
    st.subheader("**Detailed View of the Project with C4 Diagrams**")

    col1, col2 = st.columns(2)

    with st.container():

        with col1:
            st.subheader("**Context Diagram**")
            st.image("./images/Context.jpg", caption="Context of C4 Model", width=500)
        with col2:
            st.subheader("**Explanation**")
            st.write(
            "This C4 diagram provides a detailed view of the Context of the Project"

            "1. **Person (User):** Represents the end user interacting with the system.\n\n"
                
            "2. **System (Sentiment Analysis):** Represents the core of the system, which is responsible for sentiment analysis "
            "of user comments. This system analyzes comments fetched from YouTube videos.\n\n"
                
            "3. **Container (YouTube Dataset):** Represents the container holding the YouTube dataset accessed using the YouTube API. "
            "The 'User' interacts with this 'Container' to fetch videos for analysis.\n\n"
                
            "4. **Container (Output):** Represents the container where the most recognized videos based on sentiment analysis are outputted. "
            "This is the result of the sentiment analysis performed by the 'Sentiment Analysis' system.\n\n"
                
            "The relationships between these components are depicted with arrows, indicating the flow of actions:\n"
            "   - The 'User' contains and interacts with the 'YouTube Dataset' to fetch videos for analysis.\n"
            "   - The 'User' interacts with the 'Sentiment Analysis' system, which is responsible for analyzing comments.\n"
            "   - The 'Sentiment Analysis' system analyzes the comments and performs sentiment analysis.\n"
            "   - The output of the sentiment analysis is then directed to the 'Output' container, representing the most recognized videos."
            )

    st.write("\n\n")
    col3, col4 = st.columns(2)

    with st.container():

        with col3:
            st.subheader("**Container Diagram**")
            st.image("./images/Container.png", caption="Container of C4 Model", width=650)
        with col4:
            st.subheader("**Explanation**")
            st.write(
            "This C4 diagram provides a detailed view of the Project Container."

            "1. **Container (Frontend):** Represents the frontend of the system, developed using Streamlit. It's responsible for "
            "interacting with the user through a web browser.\n\n"
            
            "2. **Container (Backend):** Represents the backend of the system, developed in Python. It handles various functionalities "
            "such as processing comments, storing data, and communicating with external services.\n\n"
            
            "3. **Container (YouTube API):** Represents the external service, YouTube API, which is utilized to fetch comments from YouTube videos.\n\n"
            
            "4. **Container (Dataset):** Represents the container where comments fetched from YouTube are stored. The backend communicates "
            "with the YouTube API to fetch data and stores it in the dataset.\n\n"
            
            "5. **Container (Algorithm):** Represents the container responsible for processing comments. It utilizes Python libraries for sentiment analysis.\n\n"
            
            "6. **Container (Output):** Represents the container where the results of sentiment analysis are stored. The backend stores the results, "
            "and the frontend retrieves and displays this data to the user.\n\n"
            
            "The relationships between these containers are depicted with arrows, indicating the flow of actions and data:\n"
            "   - The 'User' interacts with the 'Frontend' through a web browser.\n"
            "   - The 'Frontend' sends requests to the 'Backend' through a REST API.\n"
            "   - The 'Backend' fetches comments from YouTube using the 'YouTube API' through API calls.\n"
            "   - The 'YouTube API' provides data to the 'Dataset' through a REST API.\n"
            "   - The 'Backend' stores comments in the 'Dataset' in a database.\n"
            "   - The 'Algorithm' processes comments using Python libraries.\n"
            "   - The results are stored in the 'Output' container by the 'Backend'.\n"
            "   - The 'Frontend' retrieves and displays data from the 'Output' through a REST API using Streamlit components."
            )

    st.write("\n\n")
    col5, col6 = st.columns(2)

    with st.container():

        with col5:
            st.subheader("**Frontent Component**")
            st.markdown("<br>" * 3, unsafe_allow_html=True)
            st.image("./images/Frontend_Component.png", caption="Frontend Component of C4 Model", width=650)

        with col6:
            st.subheader("**Explanation**")
            st.write(
            "This C4 Component diagram provides a detailed view of the components within the 'Frontend' and 'Analysis Results' containers."
            "This diagram focuses on Streamlit components and Plotly charts used in the Responsive AI system:\n\n"
            
            "1. **Container (Frontend):** Represents the main frontend container developed using Streamlit.\n\n"
            
            "2. **Container (User Interface):** Represents the user interface components within the frontend. These include:\n"
            "   - **Component (Query Input):** Streamlit text input for user queries.\n"
            "   - **Component (Video Thumbnail):** Streamlit image for displaying video thumbnails.\n"
            "   - **Component (Sentiment Summary):** Streamlit text for displaying sentiment summary.\n"
            "   - **Component (Download Link):** Streamlit button providing the option to download analysis details.\n\n"
            
            "3. **Container (Analysis Results):** Represents the components responsible for displaying sentiment analysis results. These include:\n"
            "   - **Component (Videos by Sentiment Scores):** Plotly chart for displaying videos sorted by sentiment scores.\n"
            "   - **Component (Box Plot):** Plotly chart displaying a box plot of sentiment scores by label.\n"
            "   - **Component (Histogram):** Plotly chart displaying a histogram of sentiment scores.\n"
            "   - **Component (Pie Chart):** Plotly chart displaying the distribution of sentiment labels.\n\n"
            
            "The relationships between these components are depicted with arrows, illustrating data flow and dependencies:\n"
            "   - The 'User' interacts with the 'User Interface' through a web browser.\n"
            "   - The 'Query Input' component submits queries through Streamlit events.\n"
            "   - The 'Backend' receives queries through API calls from the 'Query Input' component.\n"
            "   - The 'Backend' provides data to the 'Analysis Results' container through API responses.\n"
            "   - Various Streamlit components within 'Analysis Results' are responsible for displaying specific analysis details and charts."   
            )

    st.write("\n\n")
    col7, col8 = st.columns(2)
    
    with st.container():
        
        with col7:
            st.subheader("**Backend Component**")
            st.markdown("<br>" * 4, unsafe_allow_html=True)
            st.image("./images/Backend_Component.png", caption="Backend Component of C4 Model", width=650)

        with col8:
            st.subheader("**Explanation**")
            st.write(
            "This C4 Component diagram provides a detailed view of the components within the 'Backend' Container." 
            "1. **Container (Backend):** Represents the main backend container developed using Python.\n\n"
    
            "2. **Container (YouTube Data API):** Represents the external service responsible for interacting with the YouTube Data API. It includes:\n"
            "   - **Component (Search for Videos):** Uses the `youtube.search()` API call to search for videos based on user queries.\n"
            "   - **Component (Fetch Comments):** Uses the `youtube.commentThreads().list()` API call to fetch comments for selected videos.\n\n"
            
            "3. **Container (Analysis):** Represents the backend components responsible for sentiment analysis and data processing. It includes:\n"
            "   - **Component (Sentiment Analyzer):** Utilizes the `vaderSentiment.SentimentIntensityAnalyzer` for sentiment analysis of comments.\n"
            "   - **Component (Data Processing):** Involves data processing using the Pandas library for efficient handling and analysis.\n\n"
            
            "4. **Container (Database):** Represents the database used for storing and retrieving comments. The 'Backend' interacts with the database for data persistence and access.\n\n"
            
            "The relationships between these components are depicted with arrows, illustrating data flow and dependencies:\n"
            "   - The 'Backend' fetches data from the 'YouTube Data API' through API calls.\n"
            "   - The 'YouTube Data API' stores comments in the 'Database' for persistence.\n"
            "   - The 'Backend' processes data using the 'Analysis' components.\n"
            "   - The 'Analysis' components read and access comments from the 'Database'."
            )

    st.divider()
    col9, col10 = st.columns(2)
    
    with st.container():
        
        with col9:
            st.subheader("**Data Flow Chart**")
            st.image("./images/video_filter_flowchart.jpg", caption="Data Flow Chart", width=650)

        with col10:
            st.markdown("<br>" * 10, unsafe_allow_html=True)
            st.write("1. The user interacts with the system, initiating a request to analyze YouTube comments.")
            st.code("fetch_youtube_comments(api_key, education_query, max_results=50)")

            st.write("2. The system searches for videos based on the user's query using the YouTube Data API.")
            st.code("youtube.search().list()")

            st.write("3. For each video found, the system fetches associated comments using the YouTube Data API.")
            st.code("youtube.commentThreads().list()")

            st.write("4. The retrieved comments are processed to extract relevant information for sentiment analysis.")
            st.code("analyze_sentiment(df_comments)")

            st.write("5. Sentiment scores are calculated for each comment using VADER SentimentIntensityAnalyzer.")
            st.code("SentimentIntensityAnalyzer()")

            st.write("6. The sentiment scores are aggregated for each video.")
            st.code("calculate_final_sentiment(grouped_df)")

            st.write("7. Videos with positive sentiment are selected and sorted by positive scores.")
            st.code("selection_positive_comments_video(video_sentiment_scores)")

            st.write("8. Links to positively scored videos are generated and displayed.")
            st.code("video_sentiment_scores['link']")
    
    st.divider()

    st.subheader("**Implications for Video Recommendations**")
    st.write(
    "The integration of sentiment analysis in the recommendation system allows for more nuanced content suggestions. By considering the "
    "emotional context of user comments, the system can identify and promote videos with meaningful and diverse content, ultimately enhancing the overall user experience."
    )

    st.subheader("**Project Impact**")
    st.write(
    "Responsive AI aspires to make a lasting impact on YouTube's content recommendation landscape by addressing inherent biases "
    "and ushering in a new era of personalized, diverse, and inclusive content suggestions. By actively mitigating biases, the "
    "project aims to enhance the fairness and equity of video recommendations, ensuring that users are exposed to a broad spectrum "
    "of content. The implementation of sentiment analysis adds a layer of understanding to user engagement, enabling the system to "
    "distinguish between positive and negative sentiments in comments. This distinction allows the recommendation engine to promote "
    "videos with meaningful and constructive content, fostering a positive and enriching user experience. Additionally, by providing "
    "more personalized suggestions, Responsive AI seeks to deepen user engagement and satisfaction, tailoring recommendations to "
    "individual preferences. The ultimate goal is to create an online environment that not only reflects the diversity of content "
    "but also contributes to a more inclusive digital community."
    )

    st.subheader("**References**")
    st.write(
    "1. **Streamlit app**: https://docs.streamlit.io/ \n"
    "2. **Sentiment Analysis in Youtube**: https://medium.com/analytics-vidhya/sentiment-analysis-of-a-youtube-video-63ced6b7b1c4 \n"
    "3. **To get API**: https://console.cloud.google.com/ \n"
    "4. **About Bias and Fairness**: https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias"
    )

elif selected_nav == "Simulation":
    st.markdown(
    """
    <div style="text-align: center;">
        <h2><b>Youtube Sentiment Analysis Application<b></h2>
        <p>Simulation of the Application</p>
    </div>
    """,
    unsafe_allow_html=True,
    )

    # User input for API key
    api_key = st.text_input("Enter your YouTube API key:", type="password")

    # User query input
    user_query = st.text_input("Enter your query:")
    st.write(f"You entered: {user_query}")

    # Display embedded YouTube videos based on the user query
    if st.button("Simulate Video Query"):
        filtered_videos_df, video_sentiment_scores, grouped_df = simulate_and_display_results(api_key, user_query)

        st.subheader("**Simulated Video Query Results**")

        # Save the analysis details to a CSV file
        analysis_details_file = "data/analysis_details.csv"
        save_analysis_details(grouped_df, file_path=analysis_details_file)

        # Provide a download link for the user
        download_file(analysis_details_file)

        for i, row in filtered_videos_df.iterrows():
            with st.container():
                st.write(f"Video {i + 1}")
                thumbnail_url = f"https://img.youtube.com/vi/{row['link'].split('=')[1]}/maxresdefault.jpg"
                link = f'<a href="{row["link"]}" target="_blank"><img src="{thumbnail_url}" width="700" margin="20 auto" display="block"></a>'
                st.markdown(link, unsafe_allow_html=True)

                # Display sentiment summary
                video_id = row['link'].split('=')[1]
                positive_score = video_sentiment_scores.loc[video_id, 'positive_score']
                
                # Calculate and display percentile
                positive_percentile = percentileofscore(video_sentiment_scores['positive_score'], positive_score)
                st.write(f"**Positive Percentile Score: {positive_percentile:.2f}%**")


elif selected_nav == "Metrics":
    st.markdown(
    """
    <div style="text-align: center;">
        <h2><b>Metrics</b></h2>
        <p>Here you can find the Metrics of the Search Results</p>
    </div>
    """,
    unsafe_allow_html=True,
    )
    df = pd.read_csv("data/analysis_details.csv")
    df.fillna(0, inplace=True)
    df.drop('comment', axis=1, inplace=True)
    df = df.drop(df.columns[0], axis=1)
    st.sidebar.header("Please Filter Here: ")
    score_options = df["sentiment_label"].unique()
    sentiment = st.sidebar.multiselect(
        "Select the Sentiment:",
        options=score_options,
        default=score_options
    )
    if not sentiment:
        st.dataframe(df)
        df_selection = df
    else:
        # Filter the dataframe based on selected options
        df_selection = df.query("sentiment_label == @sentiment")
        st.dataframe(df_selection)
    # st.dataframe(df_selection)
        
    st.divider()

    st.subheader(":bar_chart: Sentiment Dashboard")
    st.markdown("##")

    #Top KPI's
    total_comments = int(df_selection['video_id'].count())
    average_sentiment_score = round(df_selection["sentiment_score"].mean(), 1)
    star_rating = ":star:" * int((average_sentiment_score*10))

    left_column, middle_column = st.columns(2)
    with left_column:
        st.subheader("Total Comments Analysed")
        st.subheader(f"{total_comments}")
    with middle_column:
        st.subheader("Average Sentiment Scores")
        st.subheader(f"{average_sentiment_score} {star_rating}")

    st.markdown("---")

    #Charts
    videos_by_sentiment = (df_selection.groupby('video_id')['sentiment_score'].sum().sort_values(ascending=False))
    fig_video_sentiment = px.bar(
        videos_by_sentiment,
        x = "sentiment_score",
        y = videos_by_sentiment.index,
        orientation="h",
        title="<b>Videos by Sentiment Scores</b>",
        color_discrete_sequence=["#008388"] * len(videos_by_sentiment),
        template="plotly_white",
    )
    st.plotly_chart(fig_video_sentiment)

    pivot_table = df_selection.pivot_table(index='video_id', columns='sentiment_label', values='sentiment_score', aggfunc='sum')

    # Box Plot
    fig_box = px.box(df_selection, x='sentiment_label', y='sentiment_score', points="all",
                 title="<b>Box Plot of Sentiment Scores by Label</b>",
                 color='sentiment_label',
                 template="plotly_white")
    st.plotly_chart(fig_box)

    #Histogram
    fig_hist = px.histogram(df_selection, x='sentiment_score', nbins=20,
                        title="<b>Histogram of Sentiment Scores</b>",
                        template="plotly_white")
    st.plotly_chart(fig_hist)

    #Pie Chart
    fig_pie = px.pie(df_selection, names='sentiment_label',
                 title="<b>Pie Chart of Sentiment Labels Distribution</b>",
                 template="plotly_white")
    st.plotly_chart(fig_pie)

    #Area chart
    chart_data = pd.DataFrame(df_selection)
    st.write("**Sentiment Analysis: Positive vs. Negative Scores**")
    st.area_chart(
        chart_data, x="sentiment_score", y=["positive_score", "negative_score"]
    )

elif selected_nav == "Contact Us":
    # Contact Us content
    st.markdown(
    """
    <div style="text-align: center;">
        <h2><b>Contact Us</b></h2>
        <p>For any inquiries, please contact us at the following email addresses</p>
    </div>
    """,
    unsafe_allow_html=True,
    )

    st.markdown(
    """
    <div style="text-align: center;">
        <p><b>General Inquiries:</b> rincismelvin25@gmail.com</p>
        <p><b>Technical Support:</b> kailashchandran99999@gmail.com</p>
        <p><b>Project Related Queries:</b> ntkarthi29@gmail.com</p>
    </div>
    """,
    unsafe_allow_html=True,
    )
    # Add more email IDs as needed

# Add smooth scrolling to selected section
st.markdown(
    f"""
    <style>
        a[name="{selected_nav.lower().replace(' ', '_')}"] {{
            visibility: hidden;
        }}
        #{selected_nav.lower().replace(' ', '_')} {{
            visibility: visible;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<a name="{selected_nav.lower().replace(" ", "_")}"></a>', unsafe_allow_html=True)
