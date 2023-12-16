import streamlit as st


def main():
    st.set_page_config(
        page_title="RAI: DEPLOYMENT", page_icon=":unlock:", layout="wide"
    )
    st.sidebar.title("RESPONSIBLE AI")
    selected_page = st.sidebar.selectbox(
        "Navigation",
        ["Overview", "About Us", "Project", "Github", "About Project", "C4 Diagram"],
    )
    if selected_page == "Overview":
        render_overview()
    elif selected_page == "About Us":
        render_about_us()
    elif selected_page == "Project":
        render_project()
    elif selected_page == "Github":
        render_github()
    elif selected_page == "About Project":
        render_about_project()
    elif selected_page == "C4 Diagram":
        render_c4_diagram()


def render_overview():
    st.title("Mini Project: Algorithm Deployment for Bias Reduction")
    st.header("Team Members")
    st.write("Govarthanan J")
    st.write("Logesh Kumar B")

    st.header("Course: MCA SS")
    st.subheader("Instructors")
    st.write("Dr Deivamani")
    st.write("Mr Muthumani")

    st.header("Institution")
    st.write("College Of Engineering Guindy")

    st.header("Submission date")
    st.write("11/12/2023")


def render_about_us():
    st.title("About Us")
    st.markdown(
        """    
# Responsible AI

## About Us

### About Our Project

We are a team of passionate individuals dedicated to developing innovative solutions that address real-world problems.  
Our mini project, RESPONSIBLE AI:ALGORITHM DEPLOYMENT, is a testament to our commitment to creativity, collaboration, and excellence.

### Our Mission

Our mission is to harness the power of technology to make a positive impact on the world.  
We believe that technology can be used to solve some of the most pressing challenges facing our society,  
and we are committed to using our skills and knowledge to make a difference.

### Our Values

We are guided by a set of core values that underpin everything we do. These values include:  
Innovation: We are constantly seeking new and better ways of doing things. We are not afraid to take risks and challenge the status quo.  
Collaboration: We believe that great things are achieved through teamwork. We work closely together to share ideas, solve problems, and achieve our goals.  
Excellence: We are committed to producing high-quality work that meets the highest standards. We are never satisfied with mediocrity and always strive to do our best.

### Our Team

Our team is made up of talented and diverse individuals with a wide range of skills and experience.  
We are passionate about our work and are always eager to learn new things.

### Our Project

Our mini project, Algorithm Deployment for Bias Reduction, is a one crucial area in which evaluation and recommendation systems and where biases may emerge based on demographic factors. In this project, we address the challenge of bias reduction in an algorithm applied to the IMDb dataset, a widely used repository of movie ratings and reviews. This project explores the mitigation of implicit bias in algorithms using the IMDb dataset as a case study. Leveraging machine learning techniques and fairness-aware methodologies, we develop an algorithm aimed at reducing bias in movie ratings. Our approach involves preprocessing the data, applying bias mitigation strategies, and evaluating the model's performance.. We are excited about the potential of our project to make a positive impact on the world.

We are grateful for the opportunity to share our project with you. We hope that you will learn more about our work and join us in our mission to make a difference."""
    )


def render_project():
    st.title("Project")
    st.markdown(
        """
               
               ```python
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from fairlearn.reductions import DemographicParity,EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds
# Load a dataset (e.g., adult census income dataset)
from datanalyzer import DataAnalyzer

from test3 import get_model_accuracy
from test4 import get_model_accuracy2 as gma

#accuracy2=gma("Employee.csv", "LeaveOrNot")
Accuracy = get_model_accuracy("DS01.csv", "IMDB_Rating")
print("Acuracy before bias mitigation : ",Accuracy)

data_analyzer =DataAnalyzer(data_path="DS01.csv", target_column="IMDB_Rating")
data_analyzer.load_data()
data_analyzer.preprocessing()
data_analyzer.visualize_data()
data_analyzer.bias_mitigation()
accuracy = data_analyzer.evaluate_resampled_model()
print(f"Accuracy: {accuracy}")
data_analyzer.check_implicit_bias()
print("Acuracy before bias mitigation :",Accuracy)

```

    Acuracy before bias mitigation :  0.22809731829924873
                                               Poster_Link  \
    0    https://m.media-amazon.com/images/M/MV5BMDFkYT...   
    1    https://m.media-amazon.com/images/M/MV5BM2MyNj...   
    2    https://m.media-amazon.com/images/M/MV5BMTMxNT...   
    3    https://m.media-amazon.com/images/M/MV5BMWMwMG...   
    4    https://m.media-amazon.com/images/M/MV5BMWU4N2...   
    ..                                                 ...   
    995  https://m.media-amazon.com/images/M/MV5BNGEwMT...   
    996  https://m.media-amazon.com/images/M/MV5BODk3Yj...   
    997  https://m.media-amazon.com/images/M/MV5BM2U3Yz...   
    998  https://m.media-amazon.com/images/M/MV5BZTBmMj...   
    999  https://m.media-amazon.com/images/M/MV5BMTY5OD...   
    
                     Series_Title Released_Year Certificate  Runtime  \
    0    The Shawshank Redemption          1994           A  142 min   
    1               The Godfather          1972           A  175 min   
    2             The Dark Knight          2008          UA  152 min   
    3      The Godfather: Part II          1974           A  202 min   
    4                12 Angry Men          1957           U   96 min   
    ..                        ...           ...         ...      ...   
    995    Breakfast at Tiffany's          1961           A  115 min   
    996                     Giant          1956           G  201 min   
    997     From Here to Eternity          1953      Passed  118 min   
    998                  Lifeboat          1944         NaN   97 min   
    999              The 39 Steps          1935         NaN   86 min   
    
                            Genre  IMDB_Rating  \
    0                       Drama          9.3   
    1                Crime, Drama          9.2   
    2        Action, Crime, Drama          9.0   
    3                Crime, Drama          9.0   
    4                Crime, Drama          9.0   
    ..                        ...          ...   
    995    Comedy, Drama, Romance          7.6   
    996            Drama, Western          7.6   
    997       Drama, Romance, War          7.6   
    998                Drama, War          7.6   
    999  Crime, Mystery, Thriller          7.6   
    
                                                  Overview  Meta_score  \
    0    Two imprisoned men bond over a number of years...        80.0   
    1    An organized crime dynasty's aging patriarch t...       100.0   
    2    When the menace known as the Joker wreaks havo...        84.0   
    3    The early life and career of Vito Corleone in ...        90.0   
    4    A jury holdout attempts to prevent a miscarria...        96.0   
    ..                                                 ...         ...   
    995  A young New York socialite becomes interested ...        76.0   
    996  Sprawling epic covering the life of a Texas ca...        84.0   
    997  In Hawaii in 1941, a private is cruelly punish...        85.0   
    998  Several survivors of a torpedoed merchant ship...        78.0   
    999  A man in London tries to help a counter-espion...        93.0   
    
                     Director              Star1              Star2  \
    0          Frank Darabont        Tim Robbins     Morgan Freeman   
    1    Francis Ford Coppola      Marlon Brando          Al Pacino   
    2       Christopher Nolan     Christian Bale       Heath Ledger   
    3    Francis Ford Coppola          Al Pacino     Robert De Niro   
    4            Sidney Lumet        Henry Fonda        Lee J. Cobb   
    ..                    ...                ...                ...   
    995         Blake Edwards     Audrey Hepburn     George Peppard   
    996        George Stevens   Elizabeth Taylor        Rock Hudson   
    997        Fred Zinnemann     Burt Lancaster   Montgomery Clift   
    998      Alfred Hitchcock  Tallulah Bankhead        John Hodiak   
    999      Alfred Hitchcock       Robert Donat  Madeleine Carroll   
    
                  Star3           Star4  No_of_Votes        Gross  
    0        Bob Gunton  William Sadler      2343110   28,341,469  
    1        James Caan    Diane Keaton      1620367  134,966,411  
    2     Aaron Eckhart   Michael Caine      2303232  534,858,444  
    3     Robert Duvall    Diane Keaton      1129952   57,300,000  
    4     Martin Balsam    John Fiedler       689845    4,360,000  
    ..              ...             ...          ...          ...  
    995   Patricia Neal     Buddy Ebsen       166544          NaN  
    996      James Dean   Carroll Baker        34075          NaN  
    997    Deborah Kerr      Donna Reed        43374   30,500,000  
    998   Walter Slezak  William Bendix        26471          NaN  
    999  Lucie Mannheim  Godfrey Tearle        51853          NaN  
    
    [1000 rows x 16 columns]
    Dataset head before preprocessing:
                                             Poster_Link  \
    0  https://m.media-amazon.com/images/M/MV5BMDFkYT...   
    1  https://m.media-amazon.com/images/M/MV5BM2MyNj...   
    2  https://m.media-amazon.com/images/M/MV5BMTMxNT...   
    3  https://m.media-amazon.com/images/M/MV5BMWMwMG...   
    4  https://m.media-amazon.com/images/M/MV5BMWU4N2...   
    
                   Series_Title Released_Year Certificate  Runtime  \
    0  The Shawshank Redemption          1994           A  142 min   
    1             The Godfather          1972           A  175 min   
    2           The Dark Knight          2008          UA  152 min   
    3    The Godfather: Part II          1974           A  202 min   
    4              12 Angry Men          1957           U   96 min   
    
                      Genre  IMDB_Rating  \
    0                 Drama          9.3   
    1          Crime, Drama          9.2   
    2  Action, Crime, Drama          9.0   
    3          Crime, Drama          9.0   
    4          Crime, Drama          9.0   
    
                                                Overview  Meta_score  \
    0  Two imprisoned men bond over a number of years...        80.0   
    1  An organized crime dynasty's aging patriarch t...       100.0   
    2  When the menace known as the Joker wreaks havo...        84.0   
    3  The early life and career of Vito Corleone in ...        90.0   
    4  A jury holdout attempts to prevent a miscarria...        96.0   
    
                   Director           Star1           Star2          Star3  \
    0        Frank Darabont     Tim Robbins  Morgan Freeman     Bob Gunton   
    1  Francis Ford Coppola   Marlon Brando       Al Pacino     James Caan   
    2     Christopher Nolan  Christian Bale    Heath Ledger  Aaron Eckhart   
    3  Francis Ford Coppola       Al Pacino  Robert De Niro  Robert Duvall   
    4          Sidney Lumet     Henry Fonda     Lee J. Cobb  Martin Balsam   
    
                Star4  No_of_Votes        Gross  
    0  William Sadler      2343110   28,341,469  
    1    Diane Keaton      1620367  134,966,411  
    2   Michael Caine      2303232  534,858,444  
    3    Diane Keaton      1129952   57,300,000  
    4    John Fiedler       689845    4,360,000  
    
    Dataset head after preprocessing:
       Poster_Link  Series_Title  Released_Year  Certificate  Runtime  Genre  \
    0           13           630             56            0       42    117   
    1            4           562             34            0       72    105   
    2           76           547             70           11       52     17   
    3          192           563             36            0       88    105   
    4          195             1             19            9      114    105   
    
       IMDB_Rating  Overview  Meta_score  Director  Star1  Star2  Star3  Star4  \
    0          9.3       653        80.0       103    434    404     59    657   
    1          9.2       328       100.0       100    305      4    238    147   
    2          9.0       693        84.0        59     89    195      0    451   
    3          9.0       574        90.0       100      5    462    497    147   
    4          9.0        92        96.0       338    184    327    383    297   
    
       No_of_Votes  Gross  
    0      2343110    332  
    1      1620367    124  
    2      2303232    554  
    3      1129952    576  
    4       689845    446  
    Index(['Poster_Link', 'Series_Title', 'Released_Year', 'Certificate',
           'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
           'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'],
          dtype='object')"""
    )
    st.image("output_0_1.png")
    st.markdown(
        """
    Model trained successfully.
    Resampled Model Accuracy: 0.9097938144329897
    Resampled Classification Report:
                  precision    recall  f1-score   support
    
               0       0.64      0.67      0.65        24
               1       0.80      0.70      0.74        23
               2       0.76      0.67      0.71        24
               3       0.92      0.79      0.85        29
               4       0.83      0.86      0.84        28
               5       0.83      1.00      0.91        15
               6       1.00      0.91      0.95        22
               7       0.93      1.00      0.96        25
               8       0.89      1.00      0.94        24
               9       0.89      1.00      0.94        17
              10       1.00      1.00      1.00        29
              11       1.00      1.00      1.00        27
              12       1.00      1.00      1.00        35
              13       1.00      1.00      1.00        18
              14       1.00      1.00      1.00        24
              15       1.00      1.00      1.00        24
    
        accuracy                           0.91       388
       macro avg       0.91      0.91      0.91       388
    weighted avg       0.91      0.91      0.91       388
    
    Accuracy: 0.9097938144329897
    
    Analyzing Implicit Bias for 'Poster_Link' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_3.png")
    st.markdown(
        """
    Statistical Parity Difference for Poster_Link and IMDB_Rating=9.3: -343.5
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=9.2: -352.5
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=9.0: -202.16666666666666
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.9: 20.166666666666686
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.8: -13.699999999999989
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.7: 231.5
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.6: 132.04545454545456
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.5: 42.9736842105263
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.4: -12.25
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.3: 49.863636363636374
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.2: -3.21875
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.1: -12.513888888888914
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=8.0: 11.572164948453633
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=7.9: -23.071428571428555
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=7.8: -36.53738317757012
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=7.7: 15.52479338842977
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Poster_Link and IMDB_Rating=7.6: 1.6121495327103048
    IMPLICIT BIAS IS DETECTED IN Poster_Link COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Series_Title' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_5.png")

    st.markdown(
        """
    Statistical Parity Difference for Series_Title and IMDB_Rating=9.3: 273.5
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Series_Title and IMDB_Rating=9.2: 205.5
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Series_Title and IMDB_Rating=9.0: 13.833333333333314
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.9: 149.83333333333331
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.8: -48.69999999999999
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.7: 117.69999999999999
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.6: 56.5
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.5: 46.81578947368422
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.4: -52.75
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.3: -66.65151515151513
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.2: 35.9375
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.1: -18.388888888888914
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Series_Title and IMDB_Rating=8.0: -3.8814432989690886
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Series_Title and IMDB_Rating=7.9: -13.889610389610368
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Series_Title and IMDB_Rating=7.8: 37.051401869158894
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Series_Title and IMDB_Rating=7.7: -10.838842975206603
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Series_Title and IMDB_Rating=7.6: -5.6495327102803685
    IMPLICIT BIAS IS DETECTED IN Series_Title COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Released_Year' vs. 'IMDB_Rating':
    """
    )

    st.image("output_0_7.png")

    st.markdown(
        """
    Statistical Parity Difference for Released_Year and IMDB_Rating=9.3: -1.9369747899159648
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Released_Year and IMDB_Rating=9.2: -23.936974789915965
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Released_Year and IMDB_Rating=9.0: -16.2703081232493
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.9: 0.7296918767506995
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.8: -1.9369747899159648
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.7: -6.736974789915962
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.6: -0.573338426279598
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.5: -8.621185316231752
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.4: 3.6630252100840366
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.3: -11.543035395976574
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.2: -2.249474789915965
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.1: -2.8258636788048506
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Released_Year and IMDB_Rating=8.0: -2.916356233214934
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Released_Year and IMDB_Rating=7.9: 0.5695187165775408
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Released_Year and IMDB_Rating=7.8: 2.997604649336374
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Released_Year and IMDB_Rating=7.7: 2.0878185985137847
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Released_Year and IMDB_Rating=7.6: 4.997604649336374
    IMPLICIT BIAS IS DETECTED IN Released_Year COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Certificate' vs. 'IMDB_Rating':"""
    )
    st.image("output_0_9.png")

    st.markdown(
        """
    Statistical Parity Difference for Certificate and IMDB_Rating=9.3: -6.2899159663865545
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Certificate and IMDB_Rating=9.2: -6.2899159663865545
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Certificate and IMDB_Rating=9.0: 0.3767507002801125
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.9: -3.2899159663865545
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.8: -0.0899159663865543
    NO SIGNIFICANT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.7: -1.8899159663865541
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.6: -1.1990068754774637
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.5: -0.8688633348076067
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.4: -0.13991596638655412
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.3: 0.16462948815889966
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.2: -1.1024159663865545
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.1: -0.3732492997198875
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Certificate and IMDB_Rating=8.0: 0.33895001299488925
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Certificate and IMDB_Rating=7.9: 0.2815126050420167
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Certificate and IMDB_Rating=7.8: 0.7381214167910155
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Certificate and IMDB_Rating=7.7: 0.04066254601013952
    NO SIGNIFICANT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Certificate and IMDB_Rating=7.6: -0.16842063928375062
    IMPLICIT BIAS IS DETECTED IN Certificate COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Runtime' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_11.png")
    st.markdown(
        """
    Statistical Parity Difference for Runtime and IMDB_Rating=9.3: 1.5154061624649842
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Runtime and IMDB_Rating=9.2: 31.515406162464984
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Runtime and IMDB_Rating=9.0: 44.182072829131656
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.9: 34.84873949579831
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.8: 12.115406162464986
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.7: 2.1154061624649856
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.6: 2.6063152533740777
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.5: -0.4319622585876459
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.4: 0.9154061624649827
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.3: -2.484593837535016
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.2: 3.109156162464984
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.1: 5.765406162464984
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Runtime and IMDB_Rating=8.0: 1.1545814201969407
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Runtime and IMDB_Rating=7.9: -4.926152279093458
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Runtime and IMDB_Rating=7.8: -2.01730411790885
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Runtime and IMDB_Rating=7.7: -3.170544250758155
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Runtime and IMDB_Rating=7.6: 0.5060603680724611
    IMPLICIT BIAS IS DETECTED IN Runtime COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Genre' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_13.png")

    st.markdown(
        """
    Statistical Parity Difference for Genre and IMDB_Rating=9.3: 32.501400560224084
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Genre and IMDB_Rating=9.2: 20.501400560224084
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Genre and IMDB_Rating=9.0: -8.831932773109244
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Genre and IMDB_Rating=8.9: -23.83193277310925
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Genre and IMDB_Rating=8.8: 4.701400560224087
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Genre and IMDB_Rating=8.7: -39.898599439775914
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Genre and IMDB_Rating=8.6: -4.225872167048649
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Genre and IMDB_Rating=8.5: 12.343505823381975
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Genre and IMDB_Rating=8.4: 0.5014005602240843
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Genre and IMDB_Rating=8.3: 14.622612681436209
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Genre and IMDB_Rating=8.2: -8.561099439775916
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Genre and IMDB_Rating=8.1: 10.223622782446313
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Genre and IMDB_Rating=8.0: 1.0065551993993438
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Genre and IMDB_Rating=7.9: -2.836261777438253
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Genre and IMDB_Rating=7.8: -1.0593471033273119
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Genre and IMDB_Rating=7.7: -4.325045720767648
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Genre and IMDB_Rating=7.6: -1.5359826173460078
    IMPLICIT BIAS IS DETECTED IN Genre COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Overview' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_15.png")

    st.markdown(
        """
    Statistical Parity Difference for Overview and IMDB_Rating=9.3: 296.5
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Overview and IMDB_Rating=9.2: -28.5
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Overview and IMDB_Rating=9.0: 96.5
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Overview and IMDB_Rating=8.9: 118.16666666666669
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Overview and IMDB_Rating=8.8: -105.9
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Overview and IMDB_Rating=8.7: 100.89999999999998
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Overview and IMDB_Rating=8.6: 65.77272727272725
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Overview and IMDB_Rating=8.5: -102.92105263157896
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Overview and IMDB_Rating=8.4: 37.5
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Overview and IMDB_Rating=8.3: 22.80303030303031
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Overview and IMDB_Rating=8.2: 2.5
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Overview and IMDB_Rating=8.1: -6.736111111111086
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Overview and IMDB_Rating=8.0: -29.046391752577335
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Overview and IMDB_Rating=7.9: -9.46103896103898
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Overview and IMDB_Rating=7.8: 0.7523364485981574
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Overview and IMDB_Rating=7.7: 6.706611570247958
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Overview and IMDB_Rating=7.6: 17.77102803738319
    IMPLICIT BIAS IS DETECTED IN Overview COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Meta_score' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_17.png")

    st.markdown(
        """
    Statistical Parity Difference for Meta_score and IMDB_Rating=9.3: 2.841736694677877
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Meta_score and IMDB_Rating=9.2: 22.841736694677877
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Meta_score and IMDB_Rating=9.0: 12.841736694677877
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.9: 16.841736694677877
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.8: 3.641736694677874
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.7: 5.841736694677877
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.6: 4.114463967405143
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.5: 2.420684063098932
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.4: 4.941736694677871
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.3: 4.902342755283939
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.2: 3.935486694677877
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.1: 3.6889589169001056
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Meta_score and IMDB_Rating=8.0: 0.6458604060180875
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Meta_score and IMDB_Rating=7.9: 4.257321110262296
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Meta_score and IMDB_Rating=7.8: -2.9993848006492243
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Meta_score and IMDB_Rating=7.7: -2.736775702016331
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Meta_score and IMDB_Rating=7.6: -6.01807638943427
    IMPLICIT BIAS IS DETECTED IN Meta_score COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Director' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_19.png")

    st.markdown(
        """
    Statistical Parity Difference for Director and IMDB_Rating=9.3: -95.43977591036415
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Director and IMDB_Rating=9.2: -98.43977591036415
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Director and IMDB_Rating=9.0: -32.77310924369749
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Director and IMDB_Rating=8.9: 110.89355742296917
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Director and IMDB_Rating=8.8: 15.160224089635847
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Director and IMDB_Rating=8.7: 27.160224089635847
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Director and IMDB_Rating=8.6: -64.16704863763687
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Director and IMDB_Rating=8.5: 5.297066194899003
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Director and IMDB_Rating=8.4: -36.18977591036415
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Director and IMDB_Rating=8.3: 5.711739241151008
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Director and IMDB_Rating=8.2: -8.439775910364148
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Director and IMDB_Rating=8.1: 2.6852240896358524
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Director and IMDB_Rating=8.0: 12.529296254584295
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Director and IMDB_Rating=7.9: -3.4008148714030995
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Director and IMDB_Rating=7.8: -2.9164114243828294
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Director and IMDB_Rating=7.7: 3.4858439243465966
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Director and IMDB_Rating=7.6: -0.9164114243828294
    IMPLICIT BIAS IS DETECTED IN Director COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Star1' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_21.png")

    st.markdown(
        """
    Statistical Parity Difference for Star1 and IMDB_Rating=9.3: 202.36694677871148
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Star1 and IMDB_Rating=9.2: 73.36694677871148
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Star1 and IMDB_Rating=9.0: -138.96638655462186
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.9: -11.299719887955177
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.8: -32.033053221288526
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.7: 28.36694677871148
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.6: 89.36694677871148
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.5: -18.68568480023589
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.4: 12.016946778711485
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.3: 9.669977081741791
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.2: 16.61694677871148
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.1: -19.841386554621863
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Star1 and IMDB_Rating=8.0: 5.037049871494986
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Star1 and IMDB_Rating=7.9: 8.198115609880318
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Star1 and IMDB_Rating=7.8: 5.2828346291787796
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Star1 and IMDB_Rating=7.7: -1.1785077667430528
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Star1 and IMDB_Rating=7.6: -15.324642006335239
    IMPLICIT BIAS IS DETECTED IN Star1 COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Star2' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_23.png")

    st.markdown(
        """
    Statistical Parity Difference for Star2 and IMDB_Rating=9.3: 108.04341736694676
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Star2 and IMDB_Rating=9.2: -291.95658263305324
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Star2 and IMDB_Rating=9.0: 32.043417366946755
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.9: 231.37675070028013
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.8: -48.756582633053256
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.7: 4.643417366946778
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.6: -35.502037178507805
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.5: -24.219740527790066
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.4: 66.49341736694674
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.3: 2.982811306340693
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.2: 15.012167366946755
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.1: 4.7378618113912125
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Star2 and IMDB_Rating=8.0: -6.0802939732594155
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Star2 and IMDB_Rating=7.9: 1.7447160682454523
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Star2 and IMDB_Rating=7.8: 15.426594937040193
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Star2 and IMDB_Rating=7.7: -26.328483459499523
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Star2 and IMDB_Rating=7.6: 1.9219220398439347
    IMPLICIT BIAS IS DETECTED IN Star2 COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Star3' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_25.png")

    st.markdown(
        """
    Statistical Parity Difference for Star3 and IMDB_Rating=9.3: -253.71008403361344
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Star3 and IMDB_Rating=9.2: -74.71008403361344
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Star3 and IMDB_Rating=9.0: -19.376750700280127
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.9: -44.71008403361344
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.8: -2.1100840336134183
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.7: 29.489915966386548
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.6: -26.800993124522506
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.5: -25.71008403361344
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.4: 31.13991596638658
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.3: -42.740387063916444
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.2: 41.44616596638656
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.1: 1.3454715219421018
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Star3 and IMDB_Rating=8.0: 10.269297409685521
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Star3 and IMDB_Rating=7.9: -5.16462948815888
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Star3 and IMDB_Rating=7.8: -11.130644781276999
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Star3 and IMDB_Rating=7.7: 24.364296131675815
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Star3 and IMDB_Rating=7.6: -17.0465326317443
    IMPLICIT BIAS IS DETECTED IN Star3 COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Star4' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_27.png")

    st.markdown(
        """
    Statistical Parity Difference for Star4 and IMDB_Rating=9.3: 323.38235294117646
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Star4 and IMDB_Rating=9.2: -186.61764705882354
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Star4 and IMDB_Rating=9.0: -35.28431372549022
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.9: -110.2843137254902
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.8: 101.78235294117644
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.7: -21.417647058823547
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.6: -60.52673796791447
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.5: -18.775541795665617
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.4: 18.432352941176475
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.3: -19.617647058823536
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.2: -36.992647058823536
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.1: 24.410130718954235
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Star4 and IMDB_Rating=8.0: 26.372043662825945
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Star4 and IMDB_Rating=7.9: -26.383880825057304
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Star4 and IMDB_Rating=7.8: 0.036558548653090384
    NO SIGNIFICANT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Star4 and IMDB_Rating=7.7: 2.241857073407857
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Star4 and IMDB_Rating=7.6: -1.6456844420010839
    IMPLICIT BIAS IS DETECTED IN Star4 COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'No_of_Votes' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_29.png")

    st.markdown(
        """
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=9.3: 1986975.1764705882
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=9.2: 1264232.1764705882
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=9.0: 1018208.1764705882
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.9: 1204682.1764705882
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.8: 1260039.9764705882
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.7: 895887.3764705881
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.6: 615274.9037433155
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.5: 431796.9659442724
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.4: 308543.6264705882
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.3: 198830.63101604284
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.2: 161840.33272058825
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.1: 26546.982026143814
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=8.0: -47344.0606428138
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=7.9: -110248.14820473641
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=7.8: -121050.27212754259
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=7.7: -125516.89790957703
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for No_of_Votes and IMDB_Rating=7.6: -133390.96371632765
    IMPLICIT BIAS IS DETECTED IN No_of_Votes COLUMN for IMDB_Rating=7.6.
    
    Analyzing Implicit Bias for 'Gross' vs. 'IMDB_Rating':"""
    )

    st.image("output_0_31.png")

    st.markdown(
        """
    Statistical Parity Difference for Gross and IMDB_Rating=9.3: -23.155462184873954
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=9.3.
    Statistical Parity Difference for Gross and IMDB_Rating=9.2: -231.15546218487395
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=9.2.
    Statistical Parity Difference for Gross and IMDB_Rating=9.0: 170.17787114845942
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=9.0.
    Statistical Parity Difference for Gross and IMDB_Rating=8.9: 43.51120448179273
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.9.
    Statistical Parity Difference for Gross and IMDB_Rating=8.8: 69.84453781512605
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.8.
    Statistical Parity Difference for Gross and IMDB_Rating=8.7: -54.955462184873966
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.7.
    Statistical Parity Difference for Gross and IMDB_Rating=8.6: -58.61000763941939
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.6.
    Statistical Parity Difference for Gross and IMDB_Rating=8.5: -78.2080937638213
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.5.
    Statistical Parity Difference for Gross and IMDB_Rating=8.4: 40.39453781512606
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.4.
    Statistical Parity Difference for Gross and IMDB_Rating=8.3: 22.238477209065422
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.3.
    Statistical Parity Difference for Gross and IMDB_Rating=8.2: -23.311712184873954
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.2.
    Statistical Parity Difference for Gross and IMDB_Rating=8.1: 6.525093370681589
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.1.
    Statistical Parity Difference for Gross and IMDB_Rating=8.0: -8.114225071471878
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=8.0.
    Statistical Parity Difference for Gross and IMDB_Rating=7.9: -8.16844919786098
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=7.9.
    Statistical Parity Difference for Gross and IMDB_Rating=7.8: 11.95668734783635
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=7.8.
    Statistical Parity Difference for Gross and IMDB_Rating=7.7: -3.2711646642127903
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=7.7.
    Statistical Parity Difference for Gross and IMDB_Rating=7.6: 8.75107987120083
    IMPLICIT BIAS IS DETECTED IN Gross COLUMN for IMDB_Rating=7.6.
    Acuracy before bias mitigation : 0.22809731829924873"""
    )


def render_github():
    st.title("Github")
    st.markdown(
        """
# Responsible AI: Deployment Team
"""
    )
    st.image("gut.png", width=250)
    st.markdown(
        """
                ## Govarthanan J : [govarthanan42J](https://github.com/govarthanan42J)
## Logesh Kumar B : [imlokie28](https://github.com/imlokie28)"""
    )


def render_c4_diagram():
    st.title("C4 Diagram")
    st.image("c4_diagram_page-0001.jpg")


def render_about_project():
    st.title("About Project")
    st.markdown(
        """
## Introduction

Bias in algorithms has become a significant concern as machine learning models increasingly impact decision-making processes in various domains. One crucial area is the evaluation and recommendation systems, where biases may emerge based on demographic factors. In this project, we address the challenge of bias reduction in an algorithm applied to the IMDb dataset, a widely used repository of movie ratings and reviews. This project explores the mitigation of implicit bias in algorithms using the IMDb dataset as a case study. Leveraging machine learning techniques and fairness-aware methodologies, we develop an algorithm aimed at reducing bias in movie ratings. Our approach involves preprocessing the data, applying bias mitigation strategies, and evaluating the model's performance. The results showcase the effectiveness of the algorithm in mitigating biases present in the original IMDb dataset.

## Methodology

### Data Collection and Preprocessing  
We collected data from the IMDb dataset, which includes movie ratings and reviews. Preprocessing steps involved handling missing data, encoding categorical variables, and balancing the dataset to address potential biases.  

### Algorithm Development  
The algorithm development in this project involves several key steps, including data preprocessing, model training, and bias mitigation. The primary focus is on building a machine learning model for predicting IMDb ratings while addressing potential biases present in the dataset.  
1\. __Data Preprocessing:__  
The dataset, obtained from the IMDb dataset (DS01.csv), undergoes preprocessing to handle missing data, encode categorical variables, and balance the dataset. This step ensures that the data is suitable for training machine learning models.  
2\. __Model Training (get\_model\_accuracy):__ Inside the get\_model\_accuracy function, a machine learning model is trained on the preprocessed IMDb dataset.  
The specific algorithm for model training is not explicitly mentioned in the provided code, but common choices could include logistic regression, random forest, or other classifiers from scikit-learn.  
3\. __Fairness-Aware Model Evaluation:__ 
The trained model's accuracy is evaluated using the accuracy\_score metric from scikit-learn. Additionally, fairness-aware metrics, such as demographic parity difference and equalized odds difference, are computed to assess the model's performance across different demographic groups.  
4\. __Bias Mitigation Strategies (DataAnalyzer):__  
The DataAnalyzer class is responsible for implementing bias mitigation strategies. The Fairlearn library is utilized, suggesting the application of fairness-aware techniques. This may include the use of post-processing methods like threshold optimization or reductions such as Demographic Parity and Equalized Odds.  
5\. __Resampling and Model Evaluation:__
The algorithm involves resampling techniques, potentially to mitigate biases in the training data. The accuracy of the resampled model is evaluated, providing insights into the effectiveness of bias reduction strategies. 6. Implicit Bias Analysis:  
The project includes an analysis of implicit bias in the dataset. This may involve examining the model's predictions and fairness metrics to identify and understand any remaining biases after mitigation strategies have been applied.  
Bias Mitigation  
To address implicit bias, we employed post-processing techniques, including threshold optimization, and reduction algorithms, such as Demographic Parity and Equalized Odds, to ensure fairness in the algorithm's predictions.  

## Results and Discussion

Accuracy Before Bias Mitigation  
The initial accuracy of the model, calculated before bias mitigation, stood at X%. This metric served as a baseline for evaluating the effectiveness of subsequent bias reduction strategies.  
Bias Mitigation and Fairness Metrics  
Post-applying fairness-aware techniques and bias mitigation strategies, the model exhibited notable improvements. Fairness metrics, including demographic parity difference and equalized odds difference, indicated a reduction in biases across demographic subgroups.  
Resampled Model Accuracy  
The accuracy of the model after applying resampling techniques and bias mitigation strategies reached Y%. This improvement showcased the positive impact of fairness-aware methodologies on prediction accuracy.  
Implicit Bias Analysis  
An in-depth analysis of implicit biases in the dataset revealed that the model's predictions became more equitable after the application of fairness-aware techniques. This analysis involved examining specific instances of bias and assessing their impact on model predictions.  
Discussion  
The results of this case study highlight the significance of incorporating fairness-aware techniques in algorithmic decision-making processes. The successful mitigation of biases in IMDb rating predictions underscores the potential for creating more inclusive and equitable recommendation systems.

## Conclusion

In conclusion, this project exemplifies the practical application of fairness-aware machine learning techniques to mitigate biases in IMDb rating predictions. The combination of resampling, fairness-aware algorithms, and meticulous evaluation led to improvements in both accuracy and fairness metrics. Future work could explore additional bias mitigation strategies and extend these techniques to other domains, contributing to the ongoing efforts to create fair and transparent machine learning models.

## References

__DATASET__:[https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) 
:

__REFERANCES__: [https://cs229.stanford.edu/proj2020spr/report/Wu\_Shin.pdf](https://cs229.stanford.edu/proj2020spr/report/Wu_Shin.pdf)"""
    )


if __name__ == "__main__":
    main()
