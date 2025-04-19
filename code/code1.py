# LinkedIn Content Creator AI Agent
# Comprehensive notebook implementation

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import re
import json
import random
from typing import List, Dict
import os
import openai
import streamlit as st
from wordcloud import WordCloud
from tqdm.notebook import tqdm

# Optional: Configure warnings
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# Database Setup
# -------------------------------------------------------------------------

def setup_database():
    """Initialize SQLite database for storing LinkedIn posts and engagement data"""
    conn = sqlite3.connect('linkedin_content.db')
    c = conn.cursor()
    
    # Create posts table
    c.execute('''
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        author TEXT,
        content TEXT,
        post_date TIMESTAMP,
        likes INTEGER,
        comments INTEGER,
        shares INTEGER,
        hashtags TEXT
    )
    ''')
    
    # Create profiles table
    c.execute('''
    CREATE TABLE IF NOT EXISTS profiles (
        id TEXT PRIMARY KEY,
        name TEXT,
        industry TEXT,
        followers INTEGER,
        post_count INTEGER,
        last_updated TIMESTAMP
    )
    ''')
    
    # Create feedback table
    c.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id TEXT,
        rating INTEGER,
        comments TEXT,
        timestamp TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print("Database setup complete!")

# -------------------------------------------------------------------------
# Data Collection & Analysis
# -------------------------------------------------------------------------

class LinkedInDataCollector:
    def __init__(self, db_path='linkedin_content.db'):
        """Initialize data collector with database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def insert_sample_data(self, count=50):
        """Insert sample LinkedIn post data for demonstration purposes"""
        profiles = [
            {'id': 'competitor_a', 'name': 'Competitor A', 'industry': 'Technology'},
            {'id': 'competitor_b', 'name': 'Competitor B', 'industry': 'Technology'},
            {'id': 'industry_leader', 'name': 'Industry Leader', 'industry': 'Technology'},
            {'id': 'your_account', 'name': 'Your Account', 'industry': 'Technology'}
        ]
        
        topics = ['AI', 'Machine Learning', 'Data Science', 'Digital Transformation', 
                 'Leadership', 'Innovation', 'Future of Work', 'Remote Work',
                 'Professional Development', 'Industry Trends']
        
        hashtags_options = [
            '#innovation #technology #future',
            '#leadership #business #success',
            '#AI #MachineLearning #DataScience',
            '#RemoteWork #FutureOfWork #Productivity',
            '#ProfessionalDevelopment #Growth #Learning',
            '#DigitalTransformation #Tech #Innovation',
            '#DataAnalytics #BigData #BusinessIntelligence',
            '#CareerAdvice #JobSearch #Networking',
            '#Entrepreneurship #Startup #Business',
            '#ProductManagement #Strategy #Development'
        ]
        
        # Sample content templates
        content_templates = [
            "Just published a new article on {topic}. The industry is changing rapidly, and it's important to stay ahead of the curve. What are your thoughts on this trend?",
            "I'm excited to announce our new {topic} initiative. This is going to revolutionize how we approach business in the digital age.",
            "5 key trends in {topic} to watch in 2023:\n\n1. Increased automation\n2. Focus on sustainability\n3. Data-driven decision making\n4. Collaborative ecosystems\n5. Customer-centric approaches",
            "Had a great conversation with industry experts about {topic} yesterday. The insights were invaluable and I'm looking forward to implementing what I learned.",
            "Question for my network: How is {topic} changing your approach to business? I'm seeing major shifts in how companies are adapting to new technologies."
        ]
        
        c = self.conn.cursor()
        
        # Insert profiles
        for profile in profiles:
            followers = random.randint(5000, 50000)
            post_count = random.randint(100, 500)
            c.execute('''
            INSERT OR REPLACE INTO profiles (id, name, industry, followers, post_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (profile['id'], profile['name'], profile['industry'], followers, post_count, datetime.now().isoformat()))
        
        # Insert sample posts
        for i in range(count):
            for profile in profiles:
                post_id = f"{profile['id']}_{i}"
                topic = random.choice(topics)
                content = random.choice(content_templates).format(topic=topic)
                hashtags = random.choice(hashtags_options)
                
                # Random date in the last 6 months
                days_ago = random.randint(1, 180)
                post_date = (datetime.now() - timedelta(days=days_ago)).replace(
                    hour=random.randint(8, 18),
                    minute=random.randint(0, 59)
                ).isoformat()
                
                # Random engagement metrics with realistic distributions
                likes = int(np.random.lognormal(4, 1))
                comments = int(likes * random.uniform(0.05, 0.2))
                shares = int(likes * random.uniform(0.02, 0.1))
                
                c.execute('''
                INSERT OR IGNORE INTO posts (id, author, content, post_date, likes, comments, shares, hashtags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (post_id, profile['id'], content, post_date, likes, comments, shares, hashtags))
        
        self.conn.commit()
        print(f"Inserted sample data for {len(profiles)} profiles, {count} posts each!")
    
    def analyze_post_patterns(self, profile_id, num_posts=50):
        """Analyze posting patterns and engagement metrics for a profile"""
        
        df = pd.read_sql(f"SELECT * FROM posts WHERE author = '{profile_id}' ORDER BY post_date DESC LIMIT {num_posts}", self.conn)
        
        if df.empty:
            return {"error": "No posts found for this profile"}
        
        # Time analysis
        df['post_datetime'] = pd.to_datetime(df['post_date'])
        df['hour_posted'] = df['post_datetime'].dt.hour
        df['day_posted'] = df['post_datetime'].dt.day_name()
        
        engagement = df['likes'] + df['comments']*3 + df['shares']*5
        df['engagement'] = engagement
        
        # Best posting times
        best_hours = df.groupby('hour_posted')['engagement'].mean().sort_values(ascending=False)
        best_days = df.groupby('day_posted')['engagement'].mean().sort_values(ascending=False)
        
        # Content analysis
        df['word_count'] = df['content'].apply(lambda x: len(x.split()))
        engagement_by_length = df.groupby(pd.cut(df['word_count'], bins=[0, 50, 100, 200, 500]))['engagement'].mean()
        
        # Hashtag analysis
        all_hashtags = []
        for tags in df['hashtags']:
            if tags:
                all_hashtags.extend(tags.split())
        
        top_hashtags = pd.Series(all_hashtags).value_counts().head(10)
        
        # Calculate engagement rate percentile
        df['engagement_rate'] = 100 * df['engagement'] / df['word_count']
        
        return {
            'optimal_posting_hours': best_hours.head(3).index.tolist(),
            'optimal_posting_days': best_days.head(3).index.tolist(),
            'ideal_post_length': engagement_by_length.idxmax(),
            'top_performing_hashtags': top_hashtags.index.tolist() if not top_hashtags.empty else [],
            'avg_engagement_rate': df['engagement_rate'].mean(),
            'engagement_percentiles': {
                '25%': df['engagement'].quantile(0.25),
                '50%': df['engagement'].quantile(0.5),
                '75%': df['engagement'].quantile(0.75),
                '90%': df['engagement'].quantile(0.9)
            },
            'post_frequency': {
                'posts_per_week': 7 * len(df) / (max(1, (df['post_datetime'].max() - df['post_datetime'].min()).days)),
                'avg_days_between_posts': (df['post_datetime'].max() - df['post_datetime'].min()).days / max(1, len(df) - 1)
            },
            'raw_data': {
                'hours': df.groupby('hour_posted')['engagement'].mean().to_dict(),
                'days': df.groupby('day_posted')['engagement'].mean().to_dict(),
                'lengths': engagement_by_length.to_dict(),
                'recent_engagement': df[['post_date', 'likes', 'comments', 'shares', 'engagement']].head(10).to_dict(orient='records')
            }
        }
    
    def compare_profiles(self, profile_ids):
        """Compare engagement metrics across multiple profiles"""
        results = {}
        
        for profile_id in profile_ids:
            results[profile_id] = self.analyze_post_patterns(profile_id)
        
        # Cross-comparison metrics
        comparison = {
            'engagement_rates': {pid: data['avg_engagement_rate'] for pid, data in results.items() if 'avg_engagement_rate' in data},
            'post_frequency': {pid: data['post_frequency']['posts_per_week'] for pid, data in results.items() if 'post_frequency' in data},
            'top_hashtags': {pid: data['top_performing_hashtags'][:3] for pid, data in results.items() if 'top_performing_hashtags' in data}
        }
        
        return {
            'individual_results': results,
            'comparison': comparison
        }
    
    def extract_content_topics(self, profile_id, num_posts=100):
        """Extract and analyze content topics from posts"""
        df = pd.read_sql(f"SELECT * FROM posts WHERE author = '{profile_id}' ORDER BY post_date DESC LIMIT {num_posts}", self.conn)
        
        if df.empty:
            return {"error": "No posts found"}
        
        # Simple keyword extraction (in a real system, this would use NLP)
        all_content = " ".join(df['content'].tolist()).lower()
        
        # Remove common words
        common_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 'that', 'this', 'are', 'as', 'at']
        for word in common_words:
            all_content = all_content.replace(f" {word} ", " ")
        
        # Count word frequencies
        words = all_content.split()
        word_freq = pd.Series(words).value_counts().head(30)
        
        # Calculate engagement for posts containing top words
        top_topics = []
        for word in word_freq.index[:10]:
            mask = df['content'].str.contains(word, case=False)
            if mask.any():
                avg_engagement = df.loc[mask, 'likes'].mean()
                top_topics.append({
                    'word': word,
                    'frequency': int(word_freq[word]),
                    'avg_engagement': avg_engagement
                })
        
        return {
            'top_topics': sorted(top_topics, key=lambda x: x['avg_engagement'], reverse=True),
            'word_frequencies': word_freq.to_dict()
        }

# -------------------------------------------------------------------------
# Content Generation Engine
# -------------------------------------------------------------------------

class LinkedInContentGenerator:
    def __init__(self, api_key=None, profile_info=None):
        """
        Initialize LinkedIn content generator
        
        Args:
            api_key: OpenAI API key (optional)
            profile_info: Information about the profile/company
        """
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key
            
        self.profile_info = profile_info or {
            'name': 'Your Company',
            'industry': 'Technology',
            'tone': 'Professional',
            'style': 'Informative'
        }
        
        self.feedback_history = []
        
    def generate_with_openai(self, topic, key_points, num_variations=2, tone="professional"):
        """Generate content using OpenAI API if key is available"""
        if not self.api_key:
            return self._generate_sample_posts(topic, key_points, num_variations, tone)
            
        try:
            prompt = f"""
            Create {num_variations} different LinkedIn posts about {topic}. 
            Include these key points: {', '.join(key_points)}
            Tone should be {tone}.
            
            Based on analysis, the ideal post length is between 150-200 words.
            Each post should end with 3-5 relevant hashtags.
            
            Format each post as complete text including hashtags.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a LinkedIn content specialist for {self.profile_info['name']} in the {self.profile_info['industry']} industry."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            
            full_content = response.choices[0].message.content
            post_sections = full_content.split("\n\n")
            
            variations = []
            for i, section in enumerate(post_sections):
                if i >= num_variations:
                    break
                    
                # Extract hashtags from the content
                content_parts = section.split("#")
                main_content = content_parts[0].strip()
                
                hashtags = []
                if len(content_parts) > 1:
                    for part in content_parts[1:]:
                        if part.strip():
                            hashtags.append("#" + part.strip().split()[0])
                
                variations.append({
                    "id": f"post_variation_{i+1}",
                    "content": main_content,
                    "hashtags": " ".join(hashtags),
                    "full_post": section.strip()
                })
                
            return variations
            
        except Exception as e:
            print(f"Error generating content with OpenAI: {str(e)}")
            return self._generate_sample_posts(topic, key_points, num_variations, tone)
    
    def _generate_sample_posts(self, topic, key_points, num_variations=2, tone="professional"):
        """Generate sample posts when API is not available"""
        post_templates = [
            # Professional tone
            """I've been thinking about {topic} lately and its implications for our industry.

{key_point_1}

But it's not just about that. We're also seeing {key_point_2}, which changes how we approach the challenges ahead.

What's your experience with {topic}? Have you found effective ways to integrate it into your work?

{hashtags}""",
            
            # Conversational tone
            """Just had a fascinating conversation about {topic} with some colleagues.

You know what really stood out to me? {key_point_1}

And that's not all - {key_point_2}

I'd love to hear your thoughts! Are you incorporating {topic} into your strategy?

{hashtags}""",
            
            # Thought leadership tone
            """The future of {topic} is being written right now, and it will transform how we do business.

My team's research has revealed: {key_point_1}

The implications are profound: {key_point_2}

Leaders who adapt early will define the next era in our industry.

What steps is your organization taking to prepare?

{hashtags}"""
        ]
        
        hashtag_templates = [
            "#Innovation #Technology #{topic} #FutureOfWork",
            "#{topic} #Leadership #Strategy #BusinessGrowth",
            "#Digital#{topic} #IndustryTrends #Professional"
        ]
        
        variations = []
        for i in range(num_variations):
            template_idx = i % len(post_templates)
            template = post_templates[template_idx]
            
            # Format the template
            topic_word = topic.replace(" ", "")
            hashtags = hashtag_templates[i % len(hashtag_templates)].replace("{topic}", topic_word)
            
            # Use at least 2 key points if available
            key_point_1 = key_points[0] if key_points else "This is changing how businesses operate"
            key_point_2 = key_points[1] if len(key_points) > 1 else "We need to adapt our strategies accordingly"
            
            post = template.format(
                topic=topic,
                key_point_1=key_point_1,
                key_point_2=key_point_2,
                hashtags=hashtags
            )
            
            variations.append({
                "id": f"post_variation_{i+1}",
                "content": post.split(hashtags)[0].strip(),
                "hashtags": hashtags,
                "full_post": post
            })
        
        return variations
        
    def collect_feedback(self, post_id, rating, feedback_text):
        """Collect feedback on generated content"""
        self.feedback_history.append({
            "post_id": post_id,
            "rating": rating,
            "feedback": feedback_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # In a real system, this would be stored in a database
        conn = sqlite3.connect('linkedin_content.db')
        c = conn.cursor()
        
        try:
            c.execute('''
            INSERT INTO feedback (post_id, rating, comments, timestamp)
            VALUES (?, ?, ?, ?)
            ''', (post_id, rating, feedback_text, datetime.now().isoformat()))
            conn.commit()
        except Exception as e:
            print(f"Error saving feedback: {str(e)}")
        finally:
            conn.close()
        
        return {"status": "success", "message": "Feedback recorded"}
    
    def get_feedback_insights(self):
        """Analyze feedback to improve future content generation"""
        if not self.feedback_history:
            return {"status": "No feedback available"}
            
        # Calculate average rating
        avg_rating = sum(item["rating"] for item in self.feedback_history) / len(self.feedback_history)
        
        # Extract common themes from feedback
        # In a real system, this would use NLP
        all_feedback = " ".join([item["feedback"] for item in self.feedback_history])
        
        positive_keywords = ['engaging', 'clear', 'informative', 'professional', 'liked', 'good']
        negative_keywords = ['boring', 'unclear', 'generic', 'too long', 'confusing']
        
        insights = []
        for keyword in positive_keywords:
            if keyword.lower() in all_feedback.lower():
                insights.append(f"Users positively mentioned '{keyword}'")
                
        for keyword in negative_keywords:
            if keyword.lower() in all_feedback.lower():
                insights.append(f"Users had concerns about content being '{keyword}'")
        
        return {
            "avg_rating": avg_rating,
            "feedback_count": len(self.feedback_history),
            "insights": insights[:5]  # Top 5 insights
        }

# -------------------------------------------------------------------------
# Visualization & Analytics
# -------------------------------------------------------------------------

def plot_posting_time_heatmap(data_collector, profile_id):
    """Create a heatmap of optimal posting times by day and hour"""
    conn = sqlite3.connect('linkedin_content.db')
    
    query = f"""
    SELECT 
        strftime('%w', post_date) as day_of_week,
        strftime('%H', post_date) as hour_of_day,
        AVG(likes + comments*3 + shares*5) as engagement
    FROM posts
    WHERE author = '{profile_id}'
    GROUP BY day_of_week, hour_of_day
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        return "No data available for heatmap"
    
    # Convert to numeric
    df['day_of_week'] = pd.to_numeric(df['day_of_week'])
    df['hour_of_day'] = pd.to_numeric(df['hour_of_day'])
    
    # Map day numbers to names (0=Sunday, 1=Monday, etc.)
    day_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
               4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    df['day_name'] = df['day_of_week'].map(day_map)
    
    # Create pivot table
    pivot = df.pivot_table(
        index='day_name', 
        columns='hour_of_day',
        values='engagement',
        aggfunc='mean'
    )
    
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(day_order)
    
    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt='.0f', linewidths=.5)
    plt.title(f'Engagement by Day and Hour for {profile_id}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    
    return plt

def plot_engagement_metrics(data_collector, profile_id, n_posts=50):
    """Plot engagement metrics over time"""
    conn = sqlite3.connect('linkedin_content.db')
    
    query = f"""
    SELECT post_date, likes, comments, shares
    FROM posts
    WHERE author = '{profile_id}'
    ORDER BY post_date DESC
    LIMIT {n_posts}
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        return "No data available for engagement metrics"
    
    # Convert to datetime
    df['post_date'] = pd.to_datetime(df['post_date'])
    df = df.sort_values('post_date')
    
    # Calculate engagement
    df['engagement'] = df['likes'] + df['comments']*3 + df['shares']*5
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['post_date'], df['engagement'], 'b-', marker='o')
    plt.title(f'Engagement Over Time for {profile_id}')
    plt.ylabel('Engagement Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['post_date'], df['likes'], 'g-', marker='o', label='Likes')
    plt.plot(df['post_date'], df['comments'], 'r-', marker='s', label='Comments')
    plt.plot(df['post_date'], df['shares'], 'y-', marker='^', label='Shares')
    plt.title('Breakdown by Metric')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return plt

def generate_word_cloud(data_collector, profile_id, n_posts=100):
    """Generate a word cloud of most common terms in posts"""
    conn = sqlite3.connect('linkedin_content.db')
    
    query = f"""
    SELECT content
    FROM posts
    WHERE author = '{profile_id}'
    ORDER BY post_date DESC
    LIMIT {n_posts}
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        return "No data available for word cloud"
    
    # Combine all content
    all_text = ' '.join(df['content'].tolist())
    
    # Remove common stopwords (this is simplified; a real implementation would use NLTK)
    stopwords = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with', 
                'that', 'this', 'are', 'as', 'at', 'be', 'by', 'have', 'it', 'or', 
                'was', 'but', 'what', 'from', 'you', 'an', 'your', 'we', 'our']
    
    for word in stopwords:
        all_text = all_text.replace(f" {word} ", " ")
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=3
    ).generate(all_text)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Most Common Terms in Posts for {profile_id}')
    plt.tight_layout()
    
    return plt

# -------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------

def streamlit_app():
    """Streamlit application for LinkedIn Content Creator AI"""
    st.set_page_config(page_title="LinkedIn Content Creator AI", layout="wide")
    
    # Initialize database if needed
    if not os.path.exists('linkedin_content.db'):
        setup_database()
        data_collector = LinkedInDataCollector()
        data_collector.insert_sample_data(count=30)
        data_collector.close()
    
    # Sidebar navigation
    st.sidebar.title("LinkedIn Content Creator AI")
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Content Generator", "Competitor Analysis", "Feedback History"]
    )
    
    # Initialize data collector
    data_collector = LinkedInDataCollector()
    
    # Get available profiles
    profiles = pd.read_sql("SELECT id, name FROM profiles", data_collector.conn)
    profile_options = {row['name']: row['id'] for _, row in profiles.iterrows()}
    
    if page == "Dashboard":
        st.title("LinkedIn Content Performance Dashboard")
        
        # Profile selection
        selected_name = st.selectbox("Select Profile", list(profile_options.keys()))
        selected_profile = profile_options[selected_name]
        
        # Date range selection
        st.write("### Time Period")
        date_range = st.date_input(
            "Select Date Range",
            value=[datetime.now() - timedelta(days=90), datetime.now()],
            max_value=datetime.now()
        )
        
        # Analyze patterns
        analysis = data_collector.analyze_post_patterns(selected_profile)
        
        if 'error' in analysis:
            st.error(analysis['error'])
        else:
            # Display key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg. Engagement Rate", f"{analysis['avg_engagement_rate']:.1f}%")
                
            with col2:
                if 'post_frequency' in analysis:
                    st.metric("Posts per Week", f"{analysis['post_frequency']['posts_per_week']:.1f}")
                
            with col3:
                if 'optimal_posting_days' in analysis and analysis['optimal_posting_days']:
                    st.metric("Best Day to Post", analysis['optimal_posting_days'][0])
                
            with col4:
                if 'top_performing_hashtags' in analysis and analysis['top_performing_hashtags']:
                    st.metric("Top Hashtag", analysis['top_performing_hashtags'][0])
            
            # Visualizations
            st.write("### Engagement Over Time")
            engagement_plot = plot_engagement_metrics(data_collector, selected_profile)
            if isinstance(engagement_plot, plt.Figure) or isinstance(engagement_plot, str):
                if isinstance(engagement_plot, str):
                    st.write(engagement_plot)
                else:
                    st.pyplot(engagement_plot)
            
            st.write("### Optimal Posting Times")
            heatmap = plot_posting_time_heatmap(data_collector, selected_profile)
            if isinstance(heatmap, plt.Figure) or isinstance(heatmap, str):
                if isinstance(heatmap, str):
                    st.write(heatmap)
                else:
                    st.pyplot(heatmap)
            
            # Content analysis
            st.write("### Content Analysis")
            topics = data_collector.extract_content_topics(selected_profile)
            
            if 'error' in topics:
                st.write(topics['error'])
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Top Performing Topics")
                    if 'top_topics' in topics and topics['top_topics']:
                        topic_df = pd.DataFrame(topics['top_topics'])
                        st.dataframe(topic_df)
                
                with col2:
                    st.write("#### Common Terms in Posts")
                    wordcloud = generate_word_cloud(data_collector, selected_profile)
                    if isinstance(wordcloud, plt.Figure):
                        st.pyplot(wordcloud)
                    else:
                        st.write(wordcloud)
    
    elif page == "Content Generator":
        st.title("AI-Powered LinkedIn Post Generator")
        
        # Initialize content generator
        api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
        content_generator = LinkedInContentGenerator(api_key=api_key if api_key else None)
        
        # Input form
        with st.form("post_generator_form"):
            topic = st.text_input("Main Topic", "Artificial Intelligence in Healthcare")
            
            key_points = st.text_area(
                "Key Points (one per line)", 
                "AI is revolutionizing patient diagnosis\nReduces administrative workload\nEthical considerations remain important"
            )
            key_points_list = [point.strip() for point in key_points.strip().split('\n') if point.strip()]
            
            tone_options = ["Professional", "Conversational", "Thought Leadership", "Inspiring"]
            tone = st.selectbox("Tone", tone_options)
            
            num_variations = st.slider("Number of Variations", 1, 3, 2)
            
            submit_button = st.form_submit_button("Generate Posts")
        
        # Generate content on form submission
        if submit_button:
            with st.spinner("Generating LinkedIn posts..."):
                variations = content_generator.generate_with_openai(
                    topic=topic,
                    key_points=key_points_list,
                    num_variations=num_variations,
                    tone=tone.lower()
                )
            
            st.success(f"Generated {len(variations)} post variations!")
            
            # Display each variation with feedback collection
            for i, variation in enumerate(variations):
                with st.expander(f"Post Variation {i+1}", expanded=i==0):
                    st.markdown(variation["full_post"])
                    
                    # Buttons for immediate actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Use This Post", key=f"use_{i}"):
                            st.session_state[f"selected_post_{i}"] = variation["full_post"]
                            st.success("Post selected! Copy it to your clipboard.")
                    
                    with col2:
                        if st.button("✏️ Edit This Post", key=f"edit_{i}"):
                            st.session_state[f"editing_post_{i}"] = True
                    
                    # Show editing interface if requested
                    if f"editing_post_{i}" in st.session_state and st.session_state[f"editing_post_{i}"]:
                        edited_post = st.text_area("Edit Post", variation["full_post"], key=f"edited_text_{i}")
                        if st.button("Save Edits", key=f"save_edits_{i}"):
                            st.session_state[f"selected_post_{i}"] = edited_post
                            st.success("Edited post saved! Copy it to your clipboard.")
                    
                    # Feedback collection
                    st.write("### Provide Feedback")
                    rating = st.slider("Rate this post (1-5)", 1, 5, 3, key=f"rating_{i}")
                    feedback = st.text_input("Feedback comments (optional)", key=f"feedback_{i}")
                    
                    if st.button("Submit Feedback", key=f"submit_feedback_{i}"):
                        feedback_result = content_generator.collect_feedback(
                            post_id=variation["id"],
                            rating=rating,
                            feedback_text=feedback
                        )
                        st.success("Thank you for your feedback! This will help improve future suggestions.")
    
    elif page == "Competitor Analysis":
        st.title("Competitor Analysis")
        
        # Select competitors to compare
        st.write("### Select Profiles to Compare")
        profile_checkboxes = {}
        
        for name, profile_id in profile_options.items():
            profile_checkboxes[profile_id] = st.checkbox(name, value=(name == "Your Account"))
        
        selected_profiles = [pid for pid, checked in profile_checkboxes.items() if checked]
        
        if len(selected_profiles) < 1:
            st.warning("Please select at least one profile to analyze.")
        else:
            # Run comparison
            with st.spinner("Analyzing profiles..."):
                comparison = data_collector.compare_profiles(selected_profiles)
            
            # Display comparison results
            st.write("### Engagement Rate Comparison")
            if 'comparison' in comparison and 'engagement_rates' in comparison['comparison']:
                engagement_data = comparison['comparison']['engagement_rates']
                
                # Create DataFrame for visualization
                engagement_df = pd.DataFrame({
                    'Profile': list(engagement_data.keys()),
                    'Engagement Rate': list(engagement_data.values())
                })
                
                # Plot bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = sns.barplot(x='Profile', y='Engagement Rate', data=engagement_df, ax=ax)
                plt.title('Engagement Rate Comparison')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Post frequency comparison
            st.write("### Posting Frequency Comparison")
            if 'comparison' in comparison and 'post_frequency' in comparison['comparison']:
                frequency_data = comparison['comparison']['post_frequency']
                
                # Create DataFrame for visualization
                frequency_df = pd.DataFrame({
                    'Profile': list(frequency_data.keys()),
                    'Posts per Week': list(frequency_data.values())
                })
                
                # Plot bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = sns.barplot(x='Profile', y='Posts per Week', data=frequency_df, ax=ax)
                plt.title('Posting Frequency Comparison')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Top hashtags comparison
            st.write("### Top Hashtags by Profile")
            if 'comparison' in comparison and 'top_hashtags' in comparison['comparison']:
                hashtag_data = comparison['comparison']['top_hashtags']
                
                # Create table
                hashtag_table = []
                for profile, hashtags in hashtag_data.items():
                    row = [profile]
                    row.extend(hashtags if hashtags else ["N/A", "N/A", "N/A"])
                    hashtag_table.append(row)
                
                hashtag_df = pd.DataFrame(hashtag_table, columns=['Profile', 'Top Hashtag 1', 'Top Hashtag 2', 'Top Hashtag 3'])
                st.dataframe(hashtag_df)
            
            # Detailed analysis for each selected profile
            for profile_id in selected_profiles:
                st.write(f"### Detailed Analysis: {profile_id}")
                
                if profile_id in comparison['individual_results']:
                    results = comparison['individual_results'][profile_id]
                    
                    if 'error' in results:
                        st.write(results['error'])
                        continue
                    
                    # Create expandable sections for each profile
                    with st.expander("View Details", expanded=False):
                        # Best posting times
                        if 'optimal_posting_hours' in results:
                            st.write(f"**Best Hours to Post:** {', '.join(str(h) for h in results['optimal_posting_hours'])}")
                        
                        if 'optimal_posting_days' in results:
                            st.write(f"**Best Days to Post:** {', '.join(results['optimal_posting_days'])}")
                        
                        if 'ideal_post_length' in results:
                            st.write(f"**Ideal Post Length:** {results['ideal_post_length']}")
                        
                        # Raw engagement data
                        if 'raw_data' in results and 'recent_engagement' in results['raw_data']:
                            st.write("**Recent Post Engagement:**") 
                            recent_df = pd.DataFrame(results['raw_data']['recent_engagement'])
                            st.dataframe(recent_df)
    
    elif page == "Feedback History":
        st.title("Content Feedback Analysis")
        
        # Initialize content generator to access feedback
        content_generator = LinkedInContentGenerator()
        
        # Get feedback data from database
        feedback_df = pd.read_sql("""
        SELECT f.post_id, f.rating, f.comments, f.timestamp, p.content
        FROM feedback f
        LEFT JOIN posts p ON f.post_id = p.id
        ORDER BY f.timestamp DESC
        LIMIT 100
        """, data_collector.conn)
        
        if feedback_df.empty:
            st.write("No feedback data available yet.")
        else:
            # Show feedback statistics
            st.write("### Feedback Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_rating = feedback_df['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.1f}/5")
            
            with col2:
                total_feedback = len(feedback_df)
                st.metric("Total Feedback", total_feedback)
            
            with col3:
                recent_feedback = feedback_df[feedback_df['timestamp'] >= (datetime.now() - timedelta(days=7)).isoformat()]
                st.metric("Last 7 Days", len(recent_feedback))
            
            # Rating distribution
            st.write("#### Rating Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='rating', data=feedback_df, ax=ax)
            plt.title('Feedback Ratings')
            st.pyplot(fig)
            
            # Recent feedback table
            st.write("#### Recent Feedback")
            st.dataframe(feedback_df[['timestamp', 'rating', 'comments', 'content']].head(10))
            
            # Feedback insights
            st.write("#### Feedback Insights")
            insights = content_generator.get_feedback_insights()
            
            if insights['feedback_count'] > 0:
                st.write(f"Based on {insights['feedback_count']} feedback items with average rating {insights['avg_rating']:.1f}:")
                
                for insight in insights['insights']:
                    st.write(f"- {insight}")
            else:
                st.write("No feedback insights available yet.")
    
    # Close database connection
    data_collector.close()

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize database if it doesn't exist
    if not os.path.exists('linkedin_content.db'):
        setup_database()
        
        # Insert sample data
        collector = LinkedInDataCollector()
        collector.insert_sample_data(count=50)
        collector.close()
    
    # Run the Streamlit app
    import streamlit as st
    streamlit_app()