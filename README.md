# 🎬 Netflix Movies and TV Shows - Data Analysis & ML

A comprehensive data science project analyzing Netflix's content catalog, including exploratory data analysis, visualizations, and machine learning to predict content types. This project reveals insights into Netflix's content strategy, popular genres, top directors, and global content distribution.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Netflix](https://img.shields.io/badge/Netflix-E50914?style=for-the-badge&logo=netflix&logoColor=white)

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Analysis](#data-analysis)
- [Visualizations](#visualizations)
- [Machine Learning Model](#machine-learning-model)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Author](#author)

---

## 🎯 Project Overview

This project provides a deep dive into Netflix's extensive catalog of movies and TV shows. By analyzing over 8,000 titles, we uncover patterns in content production, popular genres, geographic distribution, and trends over time.

**Key Objectives:**
- 📊 Analyze Netflix's content distribution (Movies vs TV Shows)
- 🌍 Identify top content-producing countries
- 📈 Track content release trends over the years
- 🎭 Discover most popular genres and ratings
- 👨‍🎬 Find top directors and their content preferences
- 🤖 Build ML model to predict content type

---

## 📊 Dataset Information

### Source
**Netflix Movies and TV Shows Dataset** from Kaggle

### Dataset Overview
- **Total Records**: 8,000+ Netflix titles
- **Data Type**: Movies and TV Shows
- **Time Period**: Content released up to 2021
- **Geographic Coverage**: Global (190+ countries)

### Features Description

| Feature | Description | Type |
|---------|-------------|------|
| **show_id** | Unique identifier for each title | String |
| **type** | Content type (Movie or TV Show) | Categorical |
| **title** | Name of the movie/show | String |
| **director** | Director(s) of the content | String |
| **cast** | Main cast members | String |
| **country** | Country/countries of production | String |
| **date_added** | Date added to Netflix | Date |
| **release_year** | Year of original release | Integer |
| **rating** | Content rating (TV-MA, PG-13, etc.) | Categorical |
| **duration** | Length (minutes for movies, seasons for shows) | String |
| **listed_in** | Genre(s)/Category | String |
| **description** | Brief synopsis | String |

### Content Ratings
- **TV-MA**: Mature Audiences
- **TV-14**: Parents Strongly Cautioned
- **TV-PG**: Parental Guidance Suggested
- **R**: Restricted
- **PG-13**: Parents Strongly Cautioned
- **PG**: Parental Guidance Suggested
- **TV-Y**: All Children
- **TV-Y7**: Directed to Older Children
- **G**: General Audiences
- **NC-17**: Adults Only

---

## ✨ Features

### Data Analysis
- ✅ **Comprehensive EDA**: In-depth exploration of Netflix catalog
- ✅ **Missing Value Handling**: Smart imputation strategies
- ✅ **Feature Engineering**: Extracted year from date_added
- ✅ **Data Cleaning**: Handled unknown directors, cast, countries

### Visualizations
- ✅ **Content Distribution**: Movies vs TV Shows comparison
- ✅ **Geographic Analysis**: Top 10 content-producing countries
- ✅ **Temporal Trends**: Content release over years
- ✅ **Genre Analysis**: Most popular Netflix genres
- ✅ **Rating Distribution**: Content ratings breakdown
- ✅ **Director Insights**: Top directors and their content types

### Machine Learning
- ✅ **Binary Classification**: Predict Movie vs TV Show
- ✅ **Random Forest Classifier**: Ensemble learning approach
- ✅ **Feature Encoding**: Label encoding and one-hot encoding
- ✅ **Model Evaluation**: Accuracy, confusion matrix, classification report

---

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/CODERGURU26/Netflix-Content-Analysis.git
cd Netflix-Content-Analysis
```

### Step 2: Install Dependencies

```bash
# Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Or install all at once
pip install pandas numpy matplotlib seaborn scikit-learn jupyter openpyxl
```

### Step 3: Download Dataset

Download the `netflix_titles.csv` from Kaggle:
- [Netflix Movies and TV Shows Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)

Place the dataset in your project directory.

---

## 💻 Usage

### Running the Jupyter Notebook

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the Analysis Notebook**
   - Navigate to `Netflix_Movies_and_TV_Shows_-_Kaggle.ipynb`
   - Update the file path to your dataset location

3. **Execute the Analysis**
   - Run cells sequentially
   - Explore visualizations
   - Train the ML model
   - Analyze results

### Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Netflix dataset
df = pd.read_csv('netflix_titles.csv')

# Basic exploration
print(df.head())
print(df.info())
print(df['type'].value_counts())
```

---

## 📈 Data Analysis

### Data Loading
```python
df = pd.read_csv('netflix_titles.csv')
print(f"Dataset shape: {df.shape}")
```

### Data Cleaning

**Handling Missing Values:**
```python
# Fill missing ratings
df['rating'] = df['rating'].fillna("Not Rated")

# Fill missing directors
df['director'] = df['director'].fillna("Unknown")

# Fill missing cast
df['cast'] = df['cast'].fillna("Unknown")

# Fill missing countries
df['country'] = df['country'].fillna("Unknown")
```

**Feature Engineering:**
```python
# Convert date_added to datetime
df['date_added'] = pd.to_datetime(df['date_added'])

# Extract year from date_added
df['year_added'] = df['date_added'].dt.year
```

### Exploratory Analysis

**Content Type Distribution:**
```python
content_distribution = df['type'].value_counts()
print(content_distribution)
# Movies: ~6,100
# TV Shows: ~2,400
```

**Top Producing Countries:**
```python
top_countries = df['country'].value_counts().head(10)
print(top_countries)
# United States dominates content production
```

**Release Year Trends:**
```python
release_trend = df['release_year'].value_counts().sort_index()
# Shows increasing content production over time
```

**Genre Distribution:**
```python
# Split genres and count
genres = df['listed_in'].str.split(', ').explode()
top_genres = genres.value_counts().head(10)
# International Movies, Dramas, Comedies are top genres
```

---

## 📊 Visualizations

### 1. Movies vs TV Shows Distribution

```python
sns.countplot(x="type", data=df)
plt.title("Movies vs TV Shows on Netflix")
plt.show()
```

**Insights:**
- Movies dominate Netflix's catalog (~70%)
- TV Shows make up ~30% of content
- Clear preference for movie content

---

### 2. Top 10 Content-Producing Countries

```python
top_countries = df['country'].value_counts().head(10)

top_countries.plot(kind='bar')
plt.title("Top 10 Countries Producing Netflix Content")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.show()
```

**Insights:**
- United States leads by significant margin
- India is second largest producer
- UK, Japan, South Korea follow
- Global content diversity

---

### 3. Content Release Trends Over Years

```python
release_trend = df['release_year'].value_counts().sort_index()

release_trend.plot(kind='line', figsize=(12, 6))
plt.title("Netflix Content Release Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Releases")
plt.grid(True)
plt.show()
```

**Insights:**
- Exponential growth in recent years
- Peak content production in 2017-2019
- Reflects Netflix's expansion strategy

---

### 4. Top 10 Netflix Genres

```python
plt.figure(figsize=(12, 6))

genres = df['listed_in'].str.split(', ').explode()
top_genres = genres.value_counts().head(10)

top_genres.plot(kind="bar")
plt.title("Top 10 Netflix Genres", fontsize=16)
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
```

**Top Genres:**
1. International Movies
2. Dramas
3. Comedies
4. International TV Shows
5. Documentaries
6. Action & Adventure
7. TV Dramas
8. Independent Movies
9. Children & Family Movies
10. Thrillers

---

### 5. Rating Distribution

```python
sns.countplot(y="rating", data=df, order=df['rating'].value_counts().index)
plt.title("Distribution of Netflix Ratings")
plt.xlabel("Count")
plt.ylabel("Rating")
plt.show()
```

**Insights:**
- TV-MA (Mature) is most common rating
- TV-14 and TV-PG are also prevalent
- Wide range of content for all age groups

---

### 6. Top 10 Directors on Netflix

```python
df_director = df[df['director'] != "Unknown"]
top_directors = df_director['director'].value_counts().head(10)

top_directors.plot(kind="bar", figsize=(12, 6))
plt.title("Top 10 Directors on Netflix")
plt.xlabel("Director")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45, ha='right')
plt.show()
```

**Insights:**
- Identifies most prolific Netflix directors
- Shows director productivity

---

### 7. Directors' Content Type Preferences

```python
top10_names = top_directors.index
df_top = df_director[df_director['director'].isin(top10_names)]

plt.figure(figsize=(15, 6))
sns.countplot(data=df_top, x='director', hue='type')
plt.title("Top Directors and Their Content Type")
plt.xlabel("Director")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Content Type')
plt.show()
```

**Insights:**
- Some directors specialize in movies
- Others focus on TV shows
- Few directors create both types

---

## 🤖 Machine Learning Model

### Objective
Predict whether a Netflix title is a **Movie** or **TV Show** based on its features.

### Features Used
- **release_year**: Year of original release
- **year_added**: Year added to Netflix
- **rating**: Content rating (one-hot encoded)
- **country**: Production country (one-hot encoded)
- **duration**: Length/number of seasons

### Target Variable
- **type**: Movie (0) or TV Show (1)

### Model Pipeline

#### 1. Data Preparation
```python
# Select relevant features
df_ml = df[['type', 'release_year', 'year_added', 'rating', 'country', 'duration']].copy()

# Remove rows with missing year_added
df_ml = df_ml.dropna(subset=['year_added'])
```

#### 2. Feature Engineering

**Label Encoding for Target:**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_ml['type'] = le.fit_transform(df_ml['type'])
# Movie = 0, TV Show = 1
```

**One-Hot Encoding for Categorical Features:**
```python
df_ml = pd.get_dummies(df_ml, columns=['rating', 'country'], drop_first=True)
```

**Duration Preprocessing:**
```python
# Extract numeric values from duration
# Movies: "90 min" → 90
# TV Shows: "2 Seasons" → 2
df_ml['duration'] = df_ml['duration'].str.extract('(\d+)').astype(float)
```

#### 3. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X = df_ml.drop('type', axis=1)
y = df_ml['type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### 4. Model Training
```python
from sklearn.ensemble import RandomForestClassifier

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)
```

#### 5. Predictions
```python
y_pred = model.predict(X_test)
```

#### 6. Model Evaluation
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Movie', 'TV Show']))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### Model Performance Metrics

**Expected Performance:**
- **Accuracy**: ~85-90%
- **Precision**: High for both classes
- **Recall**: Good balance between movies and TV shows
- **F1-Score**: Strong overall performance

**Feature Importance:**
```python
# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

**Most Important Features:**
1. Duration (strong predictor)
2. Release year
3. Year added
4. Specific ratings
5. Production countries

---

## 💡 Key Insights

### Content Strategy Insights

1. **Movie Dominance**
   - 70% of Netflix catalog is movies
   - Reflects broader entertainment consumption patterns
   - Movies easier to produce and acquire

2. **US Content Leadership**
   - United States produces most Netflix content
   - India emerging as second-largest producer
   - Growing international content library

3. **Genre Preferences**
   - International Movies and Dramas dominate
   - Strong demand for documentaries
   - Diverse genre portfolio for all audiences

4. **Content Growth**
   - Exponential increase in content since 2015
   - Peak production in recent years
   - Reflects Netflix's aggressive expansion

5. **Rating Distribution**
   - TV-MA (mature) content most common
   - Significant family-friendly content (TV-PG, PG)
   - Balanced portfolio across age groups

6. **Director Insights**
   - Some directors highly prolific on Netflix
   - Clear specialization (movies vs TV shows)
   - International directors well-represented

### Business Implications

**For Netflix:**
- Continue diversifying international content
- Invest in popular genres (Dramas, Comedies)
- Balance movie and TV show production
- Maintain broad rating distribution

**For Content Creators:**
- Understand Netflix's content preferences
- Focus on popular genres
- Consider international markets
- Create diverse rating content

**For Viewers:**
- Vast content library with 8,000+ titles
- Strong international representation
- Content for all age groups
- Growing library over time

---

## 🛠️ Technologies Used

### Core Libraries

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming language | 3.7+ |
| **Pandas** | Data manipulation | Latest |
| **NumPy** | Numerical computing | Latest |
| **Matplotlib** | Data visualization | Latest |
| **Seaborn** | Statistical visualization | Latest |
| **Scikit-learn** | Machine learning | Latest |
| **Jupyter Notebook** | Interactive development | Latest |

### Data Processing
- **Pandas**: Data loading, cleaning, transformation
- **NumPy**: Numerical operations

### Visualization
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical charts and heatmaps

### Machine Learning
- **Scikit-learn**:
  - RandomForestClassifier
  - LabelEncoder
  - train_test_split
  - Evaluation metrics

---

## 📂 Project Structure

```
Netflix-Content-Analysis/
│
├── Netflix_Movies_and_TV_Shows_-_Kaggle.ipynb
│   └── Main analysis notebook
│
├── netflix_titles.csv
│   └── Dataset (download from Kaggle)
│
├── visualizations/
│   ├── content_distribution.png
│   ├── top_countries.png
│   ├── release_trends.png
│   ├── genre_analysis.png
│   ├── rating_distribution.png
│   └── top_directors.png
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation (this file)
```

---

## 🔮 Future Enhancements

### Analysis Improvements
- [ ] Time series analysis of content additions
- [ ] Sentiment analysis of descriptions
- [ ] Actor/actress network analysis
- [ ] Genre co-occurrence analysis
- [ ] Regional content preferences

### Visualization Enhancements
- [ ] Interactive dashboards (Plotly, Dash)
- [ ] Geographic heat maps
- [ ] Timeline visualizations
- [ ] Word clouds for descriptions
- [ ] Network graphs for cast/directors

### Machine Learning Enhancements
- [ ] Multi-class genre prediction
- [ ] Content recommendation system
- [ ] Rating prediction model
- [ ] Success prediction based on features
- [ ] Natural Language Processing for descriptions
- [ ] Deep learning models

### Data Enrichment
- [ ] Scrape IMDb ratings
- [ ] Add viewership data
- [ ] Include budget information
- [ ] Add awards/nominations
- [ ] Incorporate user reviews

### Deployment
- [ ] Create web dashboard
- [ ] Build recommendation API
- [ ] Deploy interactive visualizations
- [ ] Create mobile app
- [ ] Implement real-time updates

---

## 📊 Sample Analysis Code

### Advanced Genre Analysis
```python
# Genre combinations
from itertools import combinations

# Split genres into lists
df['genre_list'] = df['listed_in'].str.split(', ')

# Find common genre pairs
genre_pairs = []
for genres in df['genre_list'].dropna():
    if len(genres) >= 2:
        genre_pairs.extend(list(combinations(genres, 2)))

# Count pairs
from collections import Counter
pair_counts = Counter(genre_pairs)
top_pairs = pd.DataFrame(pair_counts.most_common(10), 
                         columns=['Genre Pair', 'Count'])
print(top_pairs)
```

### Content Addition Patterns
```python
# Monthly content additions
df['month_added'] = df['date_added'].dt.month
df['year_month'] = df['date_added'].dt.to_period('M')

monthly_additions = df.groupby('year_month').size()

plt.figure(figsize=(15, 6))
monthly_additions.plot()
plt.title('Netflix Content Additions Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Titles Added')
plt.grid(True)
plt.show()
```

### Cast Analysis
```python
# Most frequent actors
all_cast = df['cast'].str.split(', ').explode()
top_actors = all_cast.value_counts().head(20)

plt.figure(figsize=(12, 8))
top_actors.plot(kind='barh')
plt.title('Top 20 Most Frequent Actors on Netflix')
plt.xlabel('Number of Appearances')
plt.gca().invert_yaxis()
plt.show()
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: Date Parsing Error
**Error:** Unable to parse date_added

**Solution:**
```python
# Use errors='coerce' to handle invalid dates
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Fill missing dates if needed
df['date_added'].fillna(method='ffill', inplace=True)
```

#### Issue 2: Genre Splitting Error
**Error:** Error when splitting genres

**Solution:**
```python
# Handle NaN values before splitting
genres = df['listed_in'].dropna().str.split(', ').explode()
```

#### Issue 3: Duration Preprocessing
**Error:** Cannot convert duration to numeric

**Solution:**
```python
# Extract numbers from duration
df['duration_numeric'] = df['duration'].str.extract('(\d+)').astype(float)

# Handle both minutes and seasons
df['duration_type'] = df['duration'].str.extract('(min|Season)')
```

#### Issue 4: Memory Error with One-Hot Encoding
**Error:** Too many columns after encoding

**Solution:**
```python
# Only encode top categories
top_countries = df['country'].value_counts().head(20).index
df_ml = df_ml[df_ml['country'].isin(top_countries)]

# Then apply one-hot encoding
df_ml = pd.get_dummies(df_ml, columns=['country'])
```

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

- ✅ **Data Cleaning**: Handling missing values and inconsistent data
- ✅ **Exploratory Data Analysis**: Comprehensive dataset exploration
- ✅ **Data Visualization**: Creating meaningful and insightful charts
- ✅ **Feature Engineering**: Creating new features from existing data
- ✅ **Text Processing**: Splitting and analyzing text fields
- ✅ **Machine Learning**: Binary classification with Random Forest
- ✅ **Model Evaluation**: Using multiple metrics for assessment
- ✅ **Data Storytelling**: Deriving business insights from data

---

## 🤝 Contributing

Contributions are welcome! Ways to contribute:

1. **Fork the repository**
2. **Create feature branch**
   ```bash
   git checkout -b feature/NewAnalysis
   ```
3. **Make improvements**
   - Add new visualizations
   - Try different ML models
   - Improve documentation
   - Add new analyses
4. **Commit and push**
   ```bash
   git commit -m 'Add: New feature'
   git push origin feature/NewAnalysis
   ```
5. **Open Pull Request**

### Contribution Ideas
- Build recommendation system
- Add NLP for description analysis
- Create interactive dashboard
- Implement content clustering
- Add statistical hypothesis testing
- Build time series forecasting

---
## 📧 Contact & Connect

### Author

**Gururaj Krishna Sharma**

- 📧 Email: [guruuu2468@gmail.com](mailto:guruuu2468@gmail.com)
- 💼 LinkedIn: [Gururaj Krishna Sharma](https://www.linkedin.com/in/gururaj-krishna-sharma)
- 💻 GitHub: [@CODERGURU26](https://github.com/CODERGURU26)

---

## 🌟 Acknowledgments

- **Netflix** for creating amazing content
- **Kaggle** for hosting the dataset
- **Shivam Bansal** for curating the dataset
- **Scikit-learn** team for ML tools
- **Seaborn & Matplotlib** for visualization libraries
- **Data science community** for inspiration

---

## 📚 Additional Resources

### Learn More
- [Netflix Technology Blog](https://netflixtechblog.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)

### Related Projects
- Movie recommendation systems
- Content popularity prediction
- Streaming platform analytics
- Entertainment industry analysis

### Recommended Reading
- "Streaming, Sharing, Stealing" by Michael D. Smith
- "The Netflix Effect" by Kevin McDonald
- "Python for Data Analysis" by Wes McKinney

---

## 🎯 Business Value

This analysis provides value to:

**Content Creators:**
- Understand Netflix's content preferences
- Identify trending genres
- Plan production strategies

**Marketing Teams:**
- Target appropriate demographics
- Understand content distribution
- Plan promotional campaigns

**Investors:**
- Assess Netflix's content strategy
- Understand market trends
- Evaluate growth patterns

**Data Scientists:**
- Learn EDA techniques
- Practice ML classification
- Build portfolio projects

---

**⭐ If you find this project helpful, please give it a star!**

**🔔 Watch this repository for updates and improvements!**

---

*Last Updated: February 2026*

**Happy Streaming! 🎬🍿**
