import requests
import streamlit as st
from streamlit_lottie import st_lottie


#Coding - Sentiment Analysis
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect
import pandas as pd
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
            video_title = item['snippet']['title']
            video_id = item['id']['videoId']
            video_link = f'https://www.youtube.com/watch?v={video_id}'

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
                            comments_dataset.append({'video_id': video_id, 'video_link': video_link, 'comment': comment_text})
                    except:
                        pass

            except HttpError as e:
                # Handle commentsDisabled error
                if "commentsDisabled" in str(e):
                    st.warning(f"Comments are disabled for the video: {video_title} ({video_link})")
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

#print(filtered_videos)


# Set page configuration
st.set_page_config(page_title="Responsive AI", page_icon=":graduation_cap:", layout="wide")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Navigation bar
nav_options = ["Home", "About Us", "Project", "Simulation", "Contact Us"]
selected_nav = st.sidebar.radio("Navigation", nav_options)

# --Assets--

lottie_coding = "https://lottie.host/37ba51a6-3953-4bf7-a72b-09bd2a22ab3b/yHsfbMKPn1.json"

# --Header Section--
with st.container():
    st.subheader("Welcome to the Home Page of Algorithm Development Team")
    st.title("Responsive AI")
    st.write("Analysis of Youtube Comments to provide Quality Videos to the User")
    st.write("Project Guides:")
    st.write("- Prof. Dr. Deivamani")
    st.write("- Mr. Muthumani")
    st.write("Visit our Github Page by clicking [here](https://github.com/RincisM/Responsive_AI/tree/main/Algorithm_Development)")

# --Content Section--
if selected_nav == "Home":
    # Home content
    st_lottie(lottie_coding, height=300, key="coding")

elif selected_nav == "About Us":
    # About Us content
    st.subheader("About Us")
    st.write("Here you can find information about the Algorithm Development Team.")

    # Dummy avatars and brief info
    st.image("./images/avatar.png", caption="Kailash Chandran J\nRoll: 2022179007\nEmail: u192519.kailashchandran@gmail.com")
    st.image("./images/avatar.png", caption="Rincis Melvin M\nRoll: 2022179025\nEmail: rincismelvin25@gmail.com")
    st.image("./images/avatar.png", caption="Karthikkeyan N T\nRoll: 2022179032\nEmail: ntkarthi29@gmail.com")

elif selected_nav == "Project":
    # Project content
    st.subheader("Project")
    
    # Abstract
    st.subheader("Abstract")
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

    # C4 diagram
    st.subheader("C4 Diagram")
    st.image("./images/Context.jpg", caption="Context of C4 Model")


elif selected_nav == "Simulation":
    # Simulation content
    st.subheader("YouTube Sentiment Analysis App")

    # User input for API key
    api_key = st.text_input("Enter your YouTube API key:", type="password")

    # User query input
    user_query = st.text_input("Enter your query:")
    st.write(f"You entered: {user_query}")

    # Display embedded YouTube videos based on the user query
    if st.button("Simulate Video Query"):
        df_videos = fetch_youtube_comments(api_key=api_key, education_query=user_query)
        st.subheader("Simulated Video Query Results")

        # Fetch comments for the simulated videos
        df_comments = fetch_youtube_comments(api_key=api_key, education_query=user_query)

        # Check if 'video_title' is present in df_videos
        if 'video_title' not in df_videos.columns:
            st.warning("Video titles not available in the results.")
        
        # Perform sentiment analysis for each video
        df_comments = analyze_sentiment(df_comments)

        # Display 5 containers with video details and sentiment analysis
        for i, row in df_videos.iterrows():
            with st.container():
                st.write(f"Video {i + 1}")

                # Check if 'video_title' is present in the row
                if 'video_title' in row:
                    st.write(f"Title: {row['video_title']}")
                else:
                    st.warning("Title not available")

                st.write(f"Link: {row['video_link']}")

                # Display sentiment analysis results
                st.write("Sentiment Analysis:")
                for _, comment_row in df_comments[df_comments['video_id'] == row['video_id']].iterrows():
                    st.write(f"Comment: {comment_row['comment']}")
                    st.write(f"Sentiment: {comment_row['sentiment_label']}")


elif selected_nav == "Contact Us":
    # Contact Us content
    st.subheader("Contact Us")
    st.write("For any inquiries, please contact us at the following email addresses:")

    # Provide email IDs
    st.write("1. General Inquiries: rincismelvin25@gmail.com")
    st.write("2. Technical Support: u192519.kailashchandran@gmail.com")
    st.write("3. Project Related Queries: ntkarthi29@gmail.com")
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
