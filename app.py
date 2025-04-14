import pandas as pd
from flask import Flask, request, jsonify, render_template,redirect, url_for
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from flask_cors import CORS
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import openai

app = Flask(__name__)
CORS(app)

openai.api_key = "your-openai-api-key"  # Replace with your key

# Load influencer data from Excel file
df = pd.read_excel("India_Influencers_Updated_Facebook.xlsx")

# Fill missing numeric values with 0
df[['followers', 'likes', 'comments', 'shares', 'prev_followers']] = df[['followers', 'likes', 'comments', 'shares', 'prev_followers']].fillna(0)

# Avoid division by zero
df['followers'] = df['followers'].replace(0, np.nan)
df['prev_followers'] = df['prev_followers'].replace(0, np.nan)

# Feature Engineering: Engagement Rate & Follower Growth
df['engagement_rate'] = ((df['likes'] + df['comments'] + df['shares']) / df['followers']).round(4)
df['follower_growth'] = ((df['followers'] - df['prev_followers']) / df['prev_followers']).round(4)

# Random Variations for Diversity
df['engagement_rate'] = df['engagement_rate'] * np.random.uniform(0.85, 1.15, size=len(df))
df['follower_growth'] = df['follower_growth'] * np.random.uniform(0.9, 1.1, size=len(df))

# Round again
df['engagement_rate'] = df['engagement_rate'].round(4)
df['follower_growth'] = df['follower_growth'].round(4)

# Replace NaN
df[['engagement_rate', 'follower_growth']] = df[['engagement_rate', 'follower_growth']].fillna(0)

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
df['content_sentiment'] = df['content_sentiment'].fillna('')
df['sentiment_score'] = df['content_sentiment'].apply(lambda x: round(analyzer.polarity_scores(str(x))['compound'], 4))

# Drop unnecessary columns
df.drop(columns=['prev_followers', 'content_sentiment'], inplace=True)

# Classification Labels
def classify_evaluation(score):
    if score >= 0.8:
        return "Best"
    elif score >= 0.6:
        return "Better"
    elif score >= 0.4:
        return "Good"
    elif score >= 0.2:
        return "Normal"
    elif score >= 0.1:
        return "Bad"
    else:
        return "Worst"

# Encode Labels for ML
le = LabelEncoder()
df['Evaluation'] = df['final_score'] = (0.4 * df['engagement_rate']) + (0.3 * df['follower_growth']) + (0.3 * df['sentiment_score'])
df['Evaluation'] = df['final_score'].apply(classify_evaluation)
df['Evaluation'] = le.fit_transform(df['Evaluation'])  # Convert categories to numbers

# Features and Target
X = df[['engagement_rate', 'follower_growth', 'sentiment_score']]
y = df['Evaluation']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model Accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save Model
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Load Model for API
rf_model = joblib.load('random_forest_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to Rank Influencers
def rank_influencers(filtered_df):
    for col in ['engagement_rate', 'follower_growth', 'sentiment_score']:
        if filtered_df[col].max() != filtered_df[col].min():
            filtered_df[col + '_norm'] = ((filtered_df[col] - filtered_df[col].min()) / (filtered_df[col].max() - filtered_df[col].min())).round(4)
        else:
            filtered_df[col + '_norm'] = 0 

    filtered_df['final_score'] = (0.4 * filtered_df['engagement_rate_norm']) + (0.3 * filtered_df['follower_growth_norm']) + (0.3 * filtered_df['sentiment_score_norm'])
    filtered_df['final_score'] = filtered_df['final_score'].round(4)

    filtered_df['Evaluation'] = filtered_df['final_score'].apply(classify_evaluation)
    ranked_df = filtered_df.sort_values(by='final_score', ascending=False)

    return ranked_df[['name', 'platform', 'area', 'engagement_rate', 'follower_growth', 'sentiment_score', 'final_score', 'Evaluation']]

# API to Get Influencer Data for Pie Chart
@app.route('/influencer_data/<name>', methods=['GET'])
def get_influencer_data(name):
    influencer = df[df['name'].str.lower() == name.lower()]

    if influencer.empty:
        return jsonify({"error": "Influencer not found"}), 404

    data = {
        "name": influencer.iloc[0]['name'],
        "engagement_rate": influencer.iloc[0]['engagement_rate'],
        "follower_growth": influencer.iloc[0]['follower_growth'],
        "sentiment_score": influencer.iloc[0]['sentiment_score']
    }

    return jsonify(data)

# API to Get Ranked Influencers
@app.route('/ranked_influencers', methods=['GET'])
def get_ranked_influencers():
    ranked_df = rank_influencers(df)
    return jsonify(ranked_df.to_dict(orient='records'))

# New API: Predict Influencer Evaluation Using ML
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    engagement_rate = data.get('engagement_rate', 0)
    follower_growth = data.get('follower_growth', 0)
    sentiment_score = data.get('sentiment_score', 0)

    input_features = np.array([[engagement_rate, follower_growth, sentiment_score]])

    prediction = rf_model.predict(input_features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    return jsonify({"predicted_category": predicted_label})
def get_recommended_influencers():
    recommended_df = df.copy()

    # Sort by final_score (engagement + growth + sentiment)
    recommended_df = recommended_df.sort_values(by='final_score', ascending=False)

    # Take top 10
    recommended = recommended_df[['name', 'platform', 'followers', 'likes', 'comments', 'payment']].head(10)

    return recommended.to_dict(orient='records')

@app.route('/llm_results')
def llm_results():
    platform = request.args.get("platform", "").lower()
    df_filtered = df.copy()

    # Normalize platform column
    df_filtered['platform'] = df_filtered['platform'].astype(str).str.strip().str.lower()

    # Clean payment column
    df_filtered['payment'] = pd.to_numeric(df_filtered['payment'], errors='coerce').fillna(0)

    # List of platforms you support
    valid_platforms = ["instagram", "youtube", "facebook", "threads"]


    # Filter by platform only if it's valid
    if platform in valid_platforms:
        df_filtered = df_filtered[df_filtered['platform'] == platform]

    # Sort by final_score (desc) and payment (asc)
    df_sorted = df_filtered.sort_values(by=['final_score', 'payment'], ascending=[False, True])

    # Prepare final 10 influencers
    results = df_sorted[['name', 'platform', 'followers', 'likes', 'comments', 'payment']].head(10)

    # If no results found, pass an empty list
    return render_template('llm_results.html', results=results.to_dict(orient='records'), platform=platform)

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/filter_page')
def filter_page():
    return render_template('filter.html')

@app.route('/options', methods=['GET'])
def get_options():
    return jsonify({
        "platforms": list(df['platform'].unique()),
        "areas": list(df['area'].unique())
    })

@app.route('/filter', methods=['GET'])
def filter_influencers():
    platform = request.args.get('platform', 'all')
    area = request.args.get('area', 'all')

    filtered_df = df.copy()
    if platform and platform.lower() != 'all':
        filtered_df = filtered_df[filtered_df['platform'].str.lower() == platform.lower()]
    if area and area.lower() != 'all':
        filtered_df = filtered_df[filtered_df['area'].str.lower() == area.lower()]

    if filtered_df.empty:
        return jsonify([])

    ranked_df = rank_influencers(filtered_df).head(10)
    ranked_df.insert(0, 'Rank', range(1, len(ranked_df) + 1))

    return jsonify(ranked_df.to_dict(orient='records'))

@app.route('/top_influencer')
def top_influencer():
    platform = request.args.get('platform', 'all')
    area = request.args.get('area', 'all')

    filtered_df = df.copy()
    if platform != 'all':
        filtered_df = filtered_df[filtered_df['platform'] == platform]
    if area != 'all':
        filtered_df = filtered_df[filtered_df['area'] == area]

    ranked_df = rank_influencers(filtered_df).head(10)
    ranked_df.insert(0, 'Rank', range(1, len(ranked_df) + 1))
    top_influencers = ranked_df.to_dict(orient='records')

    return render_template('top_ranked.html', top_influencer=top_influencers)

@app.route('/pie_chart/<name>', methods=['GET'])
def pie_chart(name):
    influencer = df[df['name'].str.lower() == name.lower()]

    if influencer.empty:
        return "Influencer not found", 404

    data = {
        "name": influencer.iloc[0]['name'],
        "platform": influencer.iloc[0]['platform'],
        "area": influencer.iloc[0]['area'],
        "potential_growth": influencer.iloc[0]['follower_growth'],
        "payment": influencer.iloc[0]['payment'] if 'payment' in influencer else "N/A",
        "evaluation": label_encoder.inverse_transform([influencer.iloc[0]['Evaluation']])[0],
        "likes": influencer.iloc[0]['likes'],
        "comments": influencer.iloc[0]['comments'],
        "shares": influencer.iloc[0]['shares'],
        "engagement_score": influencer.iloc[0]['engagement_rate'],
        "follower_growth": influencer.iloc[0]['follower_growth'],
        "sentiment_score": influencer.iloc[0]['sentiment_score']
    }

    return render_template('pie_chart.html', data=data)
@app.route("/thankyou")
def thankyou():
    # Get influencer name from URL parameters
    influencer_name = request.args.get('name', '')
    
    if not influencer_name:
        return redirect(url_for('home'))
    
    # Get influencer data from your dataframe
    influencer = df[df['name'].str.lower() == influencer_name.lower()]
    
    if influencer.empty:
        return redirect(url_for('home'))
    
    # Prepare data for the template
    your_data = {
        'name': influencer.iloc[0]['name'],
        'platform': influencer.iloc[0]['platform'],
        'engagement_score': influencer.iloc[0]['engagement_rate'],
        'follower_growth': influencer.iloc[0]['follower_growth'],
        'sentiment_score': influencer.iloc[0]['sentiment_score'],
        'evaluation': label_encoder.inverse_transform([influencer.iloc[0]['Evaluation']])[0]
    }
    
    return render_template("thankyou.html", data=your_data)

if __name__ == '__main__':
    app.run(debug=True)
