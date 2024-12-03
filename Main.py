import time
from datetime import datetime

import praw
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import community as community_louvain
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
print("We are here (1)")

reddit = praw.Reddit(
    client_id='vD9MYPVv1L_8CSbugD-WjQ',
    client_secret='YPtfMraTVznHTl_V_KCK_q6ER7Aw2Q',
    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
)

subreddits = ['Conservative', 'Liberal', 'The_Donald', 'KamalaHarris', 'politicaldiscussion']
keywords = ['election fraud', 'voter fraud',
            'fake news', 'misinformation', 'conspiracy',
            'scheme']


#def collect_data(subreddits, keywords, limit=1000):
#    # Initialize counters for rate limiting
#    request_count = 0
#    start_time_minute = time.time()
#    start_time_ten_minutes = time.time()
#
#    # Constants for rate limits
#    MAX_REQUESTS_PER_MINUTE = 100
#    MAX_REQUESTS_PER_TEN_MINUTES = 1000
#    SLEEP_INTERVAL = 1  # Sleep interval in seconds between requests
#
#    posts = []
#    comments = []
#
#    for subreddit_name in subreddits:
#        subreddit = reddit.subreddit(subreddit_name)
#        for keyword in keywords:
#            # Use the search function to get submissions
#            submissions = subreddit.search(keyword, limit=limit)
#            for submission in submissions:
#                # Check rate limits
#                request_count += 1
#                current_time = time.time()
#
#                # Calculate elapsed time
#                elapsed_time_minute = current_time - start_time_minute
#                elapsed_time_ten_minutes = current_time - start_time_ten_minutes
#
#                # Enforce per-minute rate limit
#                if request_count >= MAX_REQUESTS_PER_MINUTE:
#                    if elapsed_time_minute < 60:
#                        sleep_time = 60 - elapsed_time_minute
#                        print(
#                            f"Rate limit reached: Sleeping for {sleep_time:.2f} seconds to comply with per-minute limit.")
#                        time.sleep(sleep_time)
#                    # Reset counters after sleeping
#                    start_time_minute = time.time()
#                    request_count = 0
#
#                # Enforce per-10-minutes rate limit
#                if request_count >= MAX_REQUESTS_PER_TEN_MINUTES:
#                    if elapsed_time_ten_minutes < 600:
#                        sleep_time = 600 - elapsed_time_ten_minutes
#                        print(
#                            f"Rate limit reached: Sleeping for {sleep_time:.2f} seconds to comply with per-10-minutes limit.")
#                        time.sleep(sleep_time)
#                    # Reset counters after sleeping
#                    start_time_ten_minutes = time.time()
#                    request_count = 0
#
#                # Collect submission data
#                posts.append({
#                    'id': submission.id,
#                    'author': str(submission.author),
#                    'title': submission.title,
#                    'selftext': submission.selftext,
#                    'created_utc': submission.created_utc,
#                    'num_comments': submission.num_comments,
#                    'score': submission.score,
#                    'url': submission.url,
#                    'subreddit': subreddit_name
#                })
#
#                # Verbose printing
#                print(
#                    f"[{datetime.datetime.now()}] API Call: Fetched submission '{submission.id}' from subreddit '{subreddit_name}'. Requests this minute: {request_count}/{MAX_REQUESTS_PER_MINUTE}")
#
#                # Replace MoreComments to get all comments
#                submission.comments.replace_more(limit=0)
#                for comment in submission.comments.list():
#                    # Collect comment data
#                    comments.append({
#                        'id': comment.id,
#                        'author': str(comment.author),
#                        'body': comment.body,
#                        'created_utc': comment.created_utc,
#                        'score': comment.score,
#                        'parent_id': comment.parent_id,
#                        'link_id': comment.link_id,
#                        'subreddit': subreddit_name
#                    })
#
#                    # Increment request count for each comment fetched
#                    request_count += 1
#                    current_time = time.time()
#                    elapsed_time_minute = current_time - start_time_minute
#                    elapsed_time_ten_minutes = current_time - start_time_ten_minutes
#
#                    # Enforce rate limits as above
#                    if request_count >= MAX_REQUESTS_PER_MINUTE:
#                        if elapsed_time_minute < 60:
#                            sleep_time = 60 - elapsed_time_minute
#                            print(
#                                f"Rate limit reached: Sleeping for {sleep_time:.2f} seconds to comply with per-minute limit.")
#                            time.sleep(sleep_time)
#                        start_time_minute = time.time()
#                        request_count = 0
#
#                    if request_count >= MAX_REQUESTS_PER_TEN_MINUTES:
#                        if elapsed_time_ten_minutes < 600:
#                            sleep_time = 600 - elapsed_time_ten_minutes
#                            print(
#                                f"Rate limit reached: Sleeping for {sleep_time:.2f} seconds to comply with per-10-minutes limit.")
#                            time.sleep(sleep_time)
#                        start_time_ten_minutes = time.time()
#                        request_count = 0
#
#                    # Verbose printing for comments
#                    print(
#                        f"[{datetime.datetime.now()}] API Call: Fetched comment '{comment.id}' from submission '{submission.id}'. Requests this minute: {request_count}/{MAX_REQUESTS_PER_MINUTE}")
#
#                    # Optional sleep to prevent hitting rate limits too quickly
#                    time.sleep(SLEEP_INTERVAL)
#
#    posts_df = pd.DataFrame(posts)
#    comments_df = pd.DataFrame(comments)
#    return posts_df, comments_df

def collect_data(subreddits, keywords, limit=10):
    # Initialize rate limiting variables
    request_count = 0
    start_time_minute = time.time()
    start_time_ten_minutes = time.time()

    # Constants for rate limits
    MAX_REQUESTS_PER_MINUTE = 100
    MAX_REQUESTS_PER_TEN_MINUTES = 1000
    SLEEP_INTERVAL = 1  # Sleep interval in seconds between requests (adjust as needed)

    posts = []
    comments = []
    max_submissions = 10  # Adjust as needed for testing
    submissions_processed = 0

    # Start time for overall time constraint
    start_time = time.time()
    max_duration = 600  # Maximum duration in seconds (10 minutes)

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for keyword in keywords:
            submissions = subreddit.search(keyword, limit=limit)
            for submission in submissions:
                if submissions_processed >= max_submissions:
                    print("Max submissions reached. Stopping data collection.")
                    break  # Exit after processing max_submissions
                if (time.time() - start_time) > max_duration:
                    print("Time limit reached. Stopping data collection.")
                    return pd.DataFrame(posts), pd.DataFrame(comments)

                # Rate limiting checks before making API call
                request_count += 1
                current_time = time.time()
                elapsed_time_minute = current_time - start_time_minute
                elapsed_time_ten_minutes = current_time - start_time_ten_minutes

                # Enforce per-minute rate limit
                if request_count >= MAX_REQUESTS_PER_MINUTE:
                    if elapsed_time_minute < 60:
                        sleep_time = 60 - elapsed_time_minute
                        print(
                            f"Rate limit reached: Sleeping for {sleep_time:.2f} seconds to comply with per-minute limit.")
                        time.sleep(sleep_time)
                    # Reset counters after sleeping
                    start_time_minute = time.time()
                    request_count = 0

                # Enforce per-10-minutes rate limit
                if request_count >= MAX_REQUESTS_PER_TEN_MINUTES:
                    if elapsed_time_ten_minutes < 600:
                        sleep_time = 600 - elapsed_time_ten_minutes
                        print(
                            f"Rate limit reached: Sleeping for {sleep_time:.2f} seconds to comply with per-10-minutes limit.")
                        time.sleep(sleep_time)
                    # Reset counters after sleeping
                    start_time_ten_minutes = time.time()
                    request_count = 0

                submissions_processed += 1

                # Collect submission data
                posts.append({
                    'id': submission.id,
                    'author': str(submission.author),
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'created_utc': submission.created_utc,
                    'num_comments': submission.num_comments,
                    'score': submission.score,
                    'url': submission.url,
                    'subreddit': subreddit_name
                })

                # Verbose printing
                print(
                    f"[{datetime.now()}] API Call: Fetched submission '{submission.id}' from subreddit '{subreddit_name}'. Requests this minute: {request_count}/{MAX_REQUESTS_PER_MINUTE}")

                # Optional sleep to space out requests (adjust as needed)
                time.sleep(SLEEP_INTERVAL)

                # Skip fetching comments for testing
                # Uncomment and adjust the following code if you need comments
                submission.comments.replace_more(limit=1)
                for comment in submission.comments.list():
                    # Rate limiting checks before making API call
                    request_count += 1
                    current_time = time.time()
                    elapsed_time_minute = current_time - start_time_minute
                    elapsed_time_ten_minutes = current_time - start_time_ten_minutes

                    # Enforce rate limits as above
                    if request_count >= MAX_REQUESTS_PER_MINUTE:
                        if elapsed_time_minute < 60:
                            sleep_time = 60 - elapsed_time_minute
                            print(f"Rate limit reached: Sleeping for {sleep_time:.2f} seconds to comply with per-minute limit.")
                            time.sleep(sleep_time)
                        start_time_minute = time.time()
                        request_count = 0

                    if request_count >= MAX_REQUESTS_PER_TEN_MINUTES:
                        if elapsed_time_ten_minutes < 600:
                            sleep_time = 600 - elapsed_time_ten_minutes
                            print(f"Rate limit reached: Sleeping for {sleep_time:.2f} seconds to comply with per-10-minutes limit.")
                            time.sleep(sleep_time)
                        start_time_ten_minutes = time.time()
                        request_count = 0

                    # Collect comment data
                    comments.append({
                        'id': comment.id,
                        'author': str(comment.author),
                        'body': comment.body,
                        'created_utc': comment.created_utc,
                        'score': comment.score,
                        'parent_id': comment.parent_id,
                        'link_id': comment.link_id,
                        'subreddit': subreddit_name
                    })

                    # Verbose printing for comments
                    print(f"[{datetime.now()}] API Call: Fetched comment '{comment.id}' from submission '{submission.id}'. Requests this minute: {request_count}/{MAX_REQUESTS_PER_MINUTE}")

                    # Optional sleep to space out requests
                    time.sleep(SLEEP_INTERVAL)

    posts_df = pd.DataFrame(posts)
    comments_df = pd.DataFrame(comments)
    return posts_df, comments_df

subreddits = ['politicaldiscussion']
keywords = ['fraud']

# Replace your original collect_data call with:
posts_df, comments_df = collect_data(subreddits, keywords, limit=100)

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if text:
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'[@#]\S+', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = text.strip()
        # Remove stopwords
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        text = ' '.join(tokens)
        return text
    else:
        return ''


posts_df['clean_text'] = posts_df['title'] + ' ' + posts_df['selftext']
posts_df['clean_text'] = posts_df['clean_text'].apply(preprocess_text)

comments_df['clean_text'] = comments_df['body'].apply(preprocess_text)

print(posts_df.head())
print(comments_df.head())

# ----

sia = SentimentIntensityAnalyzer()
posts_df['sentiment'] = posts_df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
comments_df['sentiment'] = comments_df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])


def sentiment_label(score):
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'

posts_df['sentiment_label'] = posts_df['sentiment'].apply(sentiment_label)
comments_df['sentiment_label'] = comments_df['sentiment'].apply(sentiment_label)



# Load a pre-trained model (e.g., "mrm8488/bert-tiny-finetuned-fake-news-detection")
tokenizer = AutoTokenizer.from_pretrained('mrm8488/bert-tiny-finetuned-fake-news-detection')
model = AutoModelForSequenceClassification.from_pretrained('mrm8488/bert-tiny-finetuned-fake-news-detection')

def detect_misinformation(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction  # 0: real, 1: fake

posts_df['misinformation_label'] = posts_df['clean_text'].apply(detect_misinformation)
comments_df['misinformation_label'] = comments_df['clean_text'].apply(detect_misinformation)

# Merge posts and comments to create a unified dataset
data = pd.concat([posts_df[['author', 'id', 'subreddit']], comments_df[['author', 'parent_id', 'subreddit']]], ignore_index=True)

# Clean author names
data['author'] = data['author'].replace('None', 'Anonymous')


# Extract parent post IDs for comments
comments_df['parent_post_id'] = comments_df['link_id'].apply(lambda x: x.split('_')[1])

# Create edges based on replies
edges = comments_df[['author', 'parent_id']]
edges = edges.copy().rename(columns={'author': 'source', 'parent_id': 'target'})



# Map parent IDs to authors
post_author_dict = posts_df.set_index('id')['author'].to_dict()
comment_author_dict = comments_df.set_index('id')['author'].to_dict()

def map_target_author(target_id):
    if target_id.startswith('t3_'):  # Submission
        post_id = target_id.split('_')[1]
        return post_author_dict.get(post_id, 'Anonymous')
    elif target_id.startswith('t1_'):  # Comment
        comment_id = target_id.split('_')[1]
        return comment_author_dict.get(comment_id, 'Anonymous')
    else:
        return 'Anonymous'

edges = edges.copy()
edges['target_author'] = edges['target'].apply(map_target_author)

edges = edges[['source', 'target_author']]
edges.columns = ['source', 'target']


G = nx.DiGraph()
G.add_edges_from(edges.values)


degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
pagerank = nx.pagerank(G)

is_connected = nx.is_connected(G.to_undirected())
print(f"Is the graph connected? {is_connected}")

centrality_df = pd.DataFrame({
    'author': list(degree_centrality.keys()),
    'degree_centrality': list(degree_centrality.values()),
    'betweenness_centrality': list(betweenness_centrality.values()),
    'eigenvector_centrality': list(eigenvector_centrality.values()),
    'pagerank': list(pagerank.values())
})

influential_users = centrality_df.sort_values('pagerank', ascending=False)


# Convert to undirected graph for community detection
G_undirected = G.to_undirected()

# Compute the best partition
partition = community_louvain.best_partition(G_undirected)

# Add community information to centrality_df
centrality_df['community'] = centrality_df['author'].map(partition)

community_counts = centrality_df['community'].value_counts()


# Merge centrality measures with user data
user_data = centrality_df.merge(posts_df[['author', 'misinformation_label']], on='author', how='left')
user_data = user_data.merge(comments_df[['author', 'misinformation_label']], on='author', how='left', suffixes=('_post', '_comment'))

# Fill missing values
user_data.fillna(0, inplace=True)

# Create a target variable (e.g., whether the user spreads misinformation)
user_data['misinformation'] = user_data[['misinformation_label_post', 'misinformation_label_comment']].max(axis=1)


features = ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality', 'pagerank']
X = user_data[features]
y = user_data['misinformation']






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G_undirected, k=0.1)

# Draw nodes with community colors
communities = list(set(partition.values()))
colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
node_colors = [colors[partition[node]] for node in G_undirected.nodes()]

nx.draw_networkx_nodes(G_undirected, pos, node_size=50, node_color=node_colors, alpha=0.7)
nx.draw_networkx_edges(G_undirected, pos, alpha=0.5)
plt.title('User Interaction Network with Community Detection')
plt.axis('off')
plt.show()



sns.histplot(centrality_df['pagerank'], bins=50, kde=True)
plt.title('Distribution of PageRank Centrality')
plt.show()



sentiment_counts = posts_df['sentiment_label'].value_counts()
sentiment_counts.plot(kind='bar', title='Sentiment Distribution of Posts')
plt.show()


def analyze_viral_spread_across_communities(df):
    """Analyze how viral content spreads between communities"""
    viral_posts = df[df['score'] > df['score'].quantile(0.9)]

    community_spread = {
        'posts_per_community': viral_posts['community'].value_counts().to_dict(),
        'cross_community_spread': len(
            viral_posts[viral_posts['community'] != viral_posts['community'].mode()[0]]) / len(viral_posts)
    }
    return community_spread


def calculate_time_to_viral(df):
    """Calculate average time for posts to reach viral status"""
    viral_threshold = df['score'].quantile(0.9)
    viral_posts = df[df['score'] > viral_threshold].copy()

    # Convert timestamps
    viral_posts['time_delta'] = viral_posts['timestamp'] - viral_posts.groupby('subreddit')['timestamp'].transform(
        'min')
    return viral_posts['time_delta'].mean().total_seconds() / 3600  # Convert to hours


def analyze_spread_path(df, G):
    """Analyze the path of information spread through the network"""
    viral_posts = df[df['score'] > df['score'].quantile(0.9)]

    spread_metrics = {
        'avg_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else None,
        'viral_posts_centrality': viral_posts['author'].map(nx.degree_centrality(G)).mean()
    }
    return spread_metrics


def analyze_first_appearance_by_community(df):
    """Analyze which communities first surface certain types of content"""
    df = df.sort_values('timestamp')
    first_appearances = df.groupby('misinformation_label')['community'].first()
    return first_appearances.to_dict()


def create_comprehensive_report(posts_df, comments_df, G, analysis_results):
    """Create a comprehensive visual report of the analysis"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 30))

    # 1. Network Structure Analysis
    plt.subplot(5, 2, 1)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=20, alpha=0.6)
    plt.title("Network Structure")

    # 2. Degree Distribution (Power Law)
    plt.subplot(5, 2, 2)
    degrees = [d for n, d in G.degree()]
    plt.hist(degrees, bins=50, alpha=0.7)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    plt.yscale('log')
    plt.xscale('log')

    # 3. Community Size Distribution
    plt.subplot(5, 2, 3)
    community_sizes = pd.Series(analysis_results['community_metrics']).apply(lambda x: x['size'])
    sns.barplot(x=community_sizes.index, y=community_sizes.values)
    plt.title("Community Sizes")
    plt.xticks(rotation=45)

    # 4. Echo Chamber Analysis
    plt.subplot(5, 2, 4)
    echo_chamber_scores = pd.Series(analysis_results['community_metrics']).apply(lambda x: x['echo_chamber_score'])
    sns.barplot(x=echo_chamber_scores.index, y=echo_chamber_scores.values)
    plt.title("Echo Chamber Scores by Community")
    plt.xticks(rotation=45)

    # 5. Sentiment Analysis
    plt.subplot(5, 2, 5)
    sns.boxplot(x='community', y='sentiment', data=posts_df)
    plt.title("Sentiment Distribution by Community")
    plt.xticks(rotation=45)

    # 6. Misinformation Spread
    plt.subplot(5, 2, 6)
    sns.countplot(x='community', hue='misinformation_label', data=posts_df)
    plt.title("Misinformation Distribution by Community")
    plt.xticks(rotation=45)

    # 7. Viral Content Analysis
    plt.subplot(5, 2, 7)
    viral_posts = posts_df[posts_df['score'] > posts_df['score'].quantile(0.9)]
    sns.scatterplot(data=viral_posts, x='sentiment', y='score', hue='misinformation_label', size='score')
    plt.title("Viral Content Analysis")

    # 8. Time Series Analysis
    plt.subplot(5, 2, 8)
    posts_df['date'] = pd.to_datetime(posts_df['created_utc'], unit='s')
    daily_posts = posts_df.groupby(['date', 'misinformation_label']).size().unstack()
    daily_posts.plot(kind='line')
    plt.title("Post Volume Over Time")

    # 9. User Influence Analysis
    plt.subplot(5, 2, 9)
    influence_metrics = pd.DataFrame({
        'PageRank': nx.pagerank(G),
        'Betweenness': nx.betweenness_centrality(G)
    }).sort_values('PageRank', ascending=False).head(20)

    influence_metrics.plot(kind='bar')
    plt.title("Top 20 Users by Influence Metrics")
    plt.xticks(rotation=45)

    # 10. Small World Analysis
    plt.subplot(5, 2, 10)
    metrics = analysis_results['network_properties']['small_world']
    plt.bar(['Clustering Coefficient', 'Avg Path Length', 'Small World Coefficient'],
            [metrics['clustering'], metrics['path_length'],
             metrics.get('small_world_coefficient', 0)])
    plt.title("Small World Network Metrics")

    plt.tight_layout()
    plt.savefig('social_media_analysis_report.png', dpi=300, bbox_inches='tight')

    # Generate Summary Statistics
    summary_stats = {
        'Network Statistics': {
            'Total Users': G.number_of_nodes(),
            'Total Interactions': G.number_of_edges(),
            'Average Degree': np.mean([d for n, d in G.degree()]),
            'Network Density': nx.density(G)
        },
        'Content Statistics': {
            'Total Posts': len(posts_df),
            'Total Comments': len(comments_df),
            'Average Sentiment': posts_df['sentiment'].mean(),
            'Misinformation Rate': (posts_df['misinformation_label'] == 1).mean()
        },
        'Community Statistics': {
            'Number of Communities': len(analysis_results['community_metrics']),
            'Average Community Size': np.mean([m['size'] for m in analysis_results['community_metrics'].values()]),
            'Average Echo Chamber Score': np.mean(
                [m['echo_chamber_score'] for m in analysis_results['community_metrics'].values()])
        },
        'Viral Content Statistics': {
            'Viral Post Rate': len(viral_posts) / len(posts_df),
            'Average Time to Viral': analysis_results['content_analysis']['origin']['avg_time_to_viral'],
            'Cross-Community Spread Rate': analysis_results['content_analysis']['virality']['community_spread'][
                'cross_community_spread']
        }
    }

    return summary_stats


# Add Small World Effect Analysis
def analyze_small_world_properties(G):
    # Calculate clustering coefficient and average shortest path length
    C = nx.average_clustering(G)
    # Generate random graph for comparison
    random_graph = nx.erdos_renyi_graph(len(G), nx.density(G))
    C_rand = nx.average_clustering(random_graph)

    try:
        L = nx.average_shortest_path_length(G)
        L_rand = nx.average_shortest_path_length(random_graph)
        sigma = (C / C_rand) / (L / L_rand)  # Small-world coefficient
        return {'clustering': C, 'path_length': L, 'small_world_coefficient': sigma}
    except nx.NetworkXError:
        print("Graph is not fully connected, calculating for largest component")
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        L = nx.average_shortest_path_length(G_sub)
        return {'clustering': C, 'path_length': L, 'small_world_coefficient': None}


# Add Power Law Analysis
def analyze_power_law(G):
    degrees = [d for n, d in G.degree()]
    # Fit power law distribution
    alpha, _ = np.polyfit(np.log(sorted(degrees, reverse=True)),
                          np.log(range(1, len(degrees) + 1)), 1)
    return {'power_law_exponent': -alpha}


# Enhanced Community Analysis
def analyze_community_structure(G, partition):
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    metrics = {}
    for comm_id, nodes in communities.items():
        subgraph = G.subgraph(nodes)
        metrics[comm_id] = {
            'size': len(nodes),
            'density': nx.density(subgraph),
            'clustering': nx.average_clustering(subgraph),
            'echo_chamber_score': calculate_echo_chamber_score(subgraph)
        }
    return metrics


def calculate_echo_chamber_score(G):
    # Higher score indicates stronger echo chamber characteristics
    internal_edges = G.number_of_edges()
    external_edges = sum(1 for u in G.nodes for v in G.neighbors(u)
                         if v not in G.nodes)
    if internal_edges + external_edges == 0:
        return 0
    return internal_edges / (internal_edges + external_edges)


# Enhanced Content Analysis
def analyze_content_virality(df):
    # Define viral threshold (e.g., top 10% by engagement)
    viral_threshold = df['score'].quantile(0.9)
    df['is_viral'] = df['score'] > viral_threshold

    # Analyze characteristics of viral content
    viral_analysis = {
        'avg_length': df[df['is_viral']]['clean_text'].str.len().mean(),
        'avg_sentiment': df[df['is_viral']]['sentiment'].mean(),
        'misinformation_rate': df[df['is_viral']]['misinformation_label'].mean(),
        'community_spread': analyze_viral_spread_across_communities(df)
    }
    return viral_analysis


# Origin Analysis
def analyze_information_origin(df, G):
    # Track first appearance of each piece of content
    df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    df = df.sort_values('timestamp')

    # Analyze spread patterns from original posts
    origin_analysis = {
        'avg_time_to_viral': calculate_time_to_viral(df),
        'spread_path': analyze_spread_path(df, G),
        'community_first_appearance': analyze_first_appearance_by_community(df)
    }
    return origin_analysis


# Main analysis pipeline
def run_enhanced_analysis(posts_df, comments_df, G):
    # Network properties
    small_world_metrics = analyze_small_world_properties(G)
    power_law_metrics = analyze_power_law(G)

    # Community analysis
    partition = community_louvain.best_partition(G)
    community_metrics = analyze_community_structure(G, partition)

    # Content analysis
    virality_metrics = analyze_content_virality(posts_df)
    origin_metrics = analyze_information_origin(posts_df, G)

    return {
        'network_properties': {
            'small_world': small_world_metrics,
            'power_law': power_law_metrics
        },
        'community_metrics': community_metrics,
        'content_analysis': {
            'virality': virality_metrics,
            'origin': origin_metrics
        }
    }


def run_full_analysis(posts_df, comments_df, G):
    # Run the enhanced analysis
    analysis_results = run_enhanced_analysis(posts_df, comments_df, G)

    # Create comprehensive report
    summary_stats = create_comprehensive_report(posts_df, comments_df, G, analysis_results)

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("===================")
    for category, stats in summary_stats.items():
        print(f"\n{category}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")

    print("\nVisualization saved as 'social_media_analysis_report.png'")

    return summary_stats, analysis_results

# After all your function definitions, add this main execution block:

if __name__ == "__main__":
    # 1. Data Collection
    print("Starting data collection...")
    subreddits = ['politicaldiscussion']  # Start with one for testing
    keywords = ['fraud']
    posts_df, comments_df = collect_data(subreddits, keywords, limit=100)
    print(f"Collected {len(posts_df)} posts and {len(comments_df)} comments")

    # 2. Data Preprocessing
    print("\nPreprocessing text data...")
    posts_df['clean_text'] = posts_df['title'] + ' ' + posts_df['selftext']
    posts_df['clean_text'] = posts_df['clean_text'].apply(preprocess_text)
    comments_df['clean_text'] = comments_df['body'].apply(preprocess_text)

    # 3. Sentiment Analysis
    print("\nPerforming sentiment analysis...")
    posts_df['sentiment'] = posts_df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    comments_df['sentiment'] = comments_df['clean_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    posts_df['sentiment_label'] = posts_df['sentiment'].apply(sentiment_label)
    comments_df['sentiment_label'] = comments_df['sentiment'].apply(sentiment_label)

    # 4. Misinformation Detection
    print("\nDetecting misinformation...")
    posts_df['misinformation_label'] = posts_df['clean_text'].apply(detect_misinformation)
    comments_df['misinformation_label'] = comments_df['clean_text'].apply(detect_misinformation)

    # 5. Network Creation
    print("\nBuilding network graph...")
    # Create edges based on replies
    edges = comments_df[['author', 'parent_id']]
    edges = edges.copy().rename(columns={'author': 'source', 'parent_id': 'target'})
    edges['target_author'] = edges['target'].apply(map_target_author)
    edges = edges[['source', 'target_author']]
    edges.columns = ['source', 'target']

    G = nx.DiGraph()
    G.add_edges_from(edges.values)
    G_undirected = G.to_undirected()

    # 6. Network Analysis
    print("\nPerforming network analysis...")
    # Add timestamp column for temporal analysis
    posts_df['timestamp'] = pd.to_datetime(posts_df['created_utc'], unit='s')
    comments_df['timestamp'] = pd.to_datetime(comments_df['created_utc'], unit='s')

    # Run the full analysis
    print("\nRunning comprehensive analysis...")
    summary_stats, analysis_results = run_full_analysis(posts_df, comments_df, G)

    # 7. Save all results
    print("\nSaving results...")
    # Save network visualization
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G_undirected, k=0.1)
    partition = community_louvain.best_partition(G_undirected)
    communities = list(set(partition.values()))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
    node_colors = [colors[partition[node]] for node in G_undirected.nodes()]
    nx.draw_networkx_nodes(G_undirected, pos, node_size=50, node_color=node_colors, alpha=0.7)
    nx.draw_networkx_edges(G_undirected, pos, alpha=0.5)
    plt.title('User Interaction Network with Community Detection')
    plt.axis('off')
    plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')

    # Save DataFrames
    posts_df.to_csv('analyzed_posts.csv', index=False)
    comments_df.to_csv('analyzed_comments.csv', index=False)

    print("\nAnalysis complete! Check the following files:")
    print("1. social_media_analysis_report.png - Comprehensive visualization report")
    print("2. network_visualization.png - Network structure visualization")
    print("3. analyzed_posts.csv - Processed posts data")
    print("4. analyzed_comments.csv - Processed comments data")