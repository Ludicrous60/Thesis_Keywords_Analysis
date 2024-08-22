from flask import Flask, render_template, redirect, url_for, flash, request
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import io
import base64
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv, math
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fite.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')


# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password == confirm_password:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            flash('Your account has been created! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match. Please try again.', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))



def analyze_seo(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Example SEO metrics
    title = soup.title.string if soup.title else 'No title'
    meta_description = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_description['content'] if meta_description else 'No meta description'
    
    h1_tags = [h1.text for h1 in soup.find_all('h1')]
    h2_tags = [h2.text for h2 in soup.find_all('h2')]
    
    # Keyword analysis
    text = soup.get_text()
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases_with_scores()
    
    # Recommendation model (Logistic Regression as an example)
    X = np.array([[len(meta_description), len(h1_tags), len(h2_tags)]])
    model = LogisticRegression()
    model.fit(np.array([[50, 2, 3], [120, 5, 10], [30, 1, 1]]), [1, 0, 1])  # Dummy training data
    recommendation = "Good" if model.predict(X)[0] == 1 else "Needs Improvement"
    
    return {
        'title': title,
        'meta_description': meta_description,
        'h1_tags': h1_tags,
        'h2_tags': h2_tags,
        'keywords': keywords,
        'recommendation': recommendation
    }

def create_visualizations(data):
    h1_count = len(data['h1_tags'])
    h2_count = len(data['h2_tags'])
    keywords = [kw[1] for kw in data['keywords']]
    keyword_scores = [kw[0] for kw in data['keywords']]
    
    # Create a new figure with three subplots
    fig, ax = plt.subplots(3, 1, figsize=(10, 18))
    
    # Bar chart
    ax[0].bar(['H1 Tags', 'H2 Tags'], [h1_count, h2_count], color=['#ff5733', '#33c4ff'])
    ax[0].set_ylabel('Count')
    ax[0].set_title('SEO Tag Count')
    
    # Pie chart for keyword distribution
    if len(keywords) > 0:
        ax[1].pie(keyword_scores, labels=keywords, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(keywords))))
        ax[1].set_title('Keyword Distribution')
    else:
        ax[1].text(0.5, 0.5, 'No keywords found', horizontalalignment='center', verticalalignment='center', fontsize=12)

    # Histogram of keyword scores
    ax[2].hist(keyword_scores, bins=10, color='#ff5733')
    ax[2].set_xlabel('Keyword Score')
    ax[2].set_ylabel('Frequency')
    ax[2].set_title('Keyword Score Histogram')

    # Adjust layout
    plt.tight_layout()
    
    # Save and encode image
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

def generate_optimization_tips(seo_data):
    tips = []
    
    if len(seo_data['h1_tags']) == 0:
        tips.append("Add at least one H1 tag to improve SEO.")
    if len(seo_data['meta_description']) < 50:
        tips.append("Your meta description is too short. Consider expanding it.")
    if len(seo_data['keywords']) < 5:
        tips.append("Increase the number of keywords to cover more SEO opportunities.")
    
    return tips

def keyword_density(text):
    words = word_tokenize(text)
    total_words = len(words)
    word_freq = nltk.FreqDist(words)
    density = {word: freq / total_words * 100 for word, freq in word_freq.items()}
    return sorted(density.items(), key=lambda x: x[1], reverse=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url')
        if not url:
            return "Error: No URL provided", 400
        
        try:
            seo_data = analyze_seo(url)
        except Exception as e:
            return f"Error analyzing SEO: {str(e)}", 500
        
        try:
            plot_url = create_visualizations(seo_data)
        except Exception as e:
            return f"Error creating visualizations: {str(e)}", 500
        
        try:
            tips = generate_optimization_tips(seo_data)
        except Exception as e:
            return f"Error generating optimization tips: {str(e)}", 500
        
        return render_template('index.html', seo_data=seo_data, plot_url=plot_url, tips=tips)
    return render_template('index.html')

# Ensure you download the stopwords and punkt once
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text
    words = nltk.word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    
    # Adding custom stop words
    custom_stop_words = set([
        'browser', 'javascript', 'enabled', 'cookies', 'privacy', 
        'web', 'website', 'traffic', 'update', 'latest', 'version',
        'please', 'make', 'access', 'fiverr', 'details', 'request',
        'pxcr10002539', 'ip', '27346782', 'ad', 'blockers', 'vpn', 
        'vpns', 'loading', 'fixes', 'challenge', 'disable', 'includes',
        'well', 'get', 'right', 'back', 'interfering', 'outdated', 
        'corrupt', 'data', 'issues', 'cause', 'operatesmake', 'task',
        'touchit', 'touchcomplete', 'needs', 'sure', 'cache', 'extensions',
        'human'
    ])
    
    all_stop_words = stop_words.union(custom_stop_words)
    
    filtered_words = [word for word in words if word not in all_stop_words and len(word) > 2]
    
    return filtered_words

def keyword_density(text):
    words = clean_text(text)
    total_words = len(words)
    
    # Count the frequency of each word
    word_freq = Counter(words)
    
    # Calculate density
    density = {word: (freq / total_words) * 100 for word, freq in word_freq.items()}
    
    # Sort by density, highest first
    sorted_density = sorted(density.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_density

@app.route('/keyword-density', methods=['GET', 'POST'])
def keyword_density_route():
    if request.method == 'POST':
        url = request.form['url']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content from the page
        text = soup.get_text()
        
        # Calculate keyword density
        density = keyword_density(text)
        
        return render_template('keyword_density.html', density=density)
    
    return render_template('keyword_density.html', density=[])



@app.route('/backlink-analyze', methods=['GET', 'POST'])
def backlink_analyze_route():
    if request.method == 'POST':
        url = request.form.get('url')
        if not url:
            return "Error: No URL provided", 400
        
        if not url.startswith('http'):
            url = 'http://' + url
        
        # Headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 403:
                return "Error fetching URL: Access forbidden (403)", 403
            elif response.status_code != 200:
                return f"Error fetching URL: Status code {response.status_code}", 500
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            backlinks = []
            for anchor in soup.find_all('a', href=True):
                href = anchor['href']
                text = anchor.get_text(strip=True)
                
                link_type = "external" if href.startswith('http') and not href.startswith(url) else "internal"
                rel_attribute = anchor.get('rel')
                link_rel_type = "nofollow" if rel_attribute and "nofollow" in rel_attribute else "dofollow"
                
                backlinks.append({
                    "source": href,
                    "anchor_text": text,
                    "link_type": link_type,
                    "link_rel_type": link_rel_type
                })
            
            if not backlinks:
                return "No backlinks found. The URL may not contain any anchor tags.", 404
            
            return render_template('backlink_analyze.html', backlinks=backlinks)
        
        except requests.RequestException as e:
            return f"Error fetching URL: {str(e)}", 500
        except Exception as e:
            return f"Error processing content: {str(e)}", 500
    
    return render_template('backlink_analyze.html', backlinks=[])

@app.route('/keyword-planner', methods=['GET', 'POST'])
def keyword_planner():
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        if not csv_file:
            return "Error: No CSV file provided", 400

        keyword_list = []
        try:
            # Open the CSV file in text mode
            csv_reader = csv.reader(io.TextIOWrapper(csv_file, encoding='utf-8'), delimiter=',')
            for row in csv_reader:
                keyword_list.append(row[0])
        except Exception as e:
            return f"Error reading CSV file: {str(e)}", 500

        try:
            # Simulated data processing without actual Google Ads API
            keyword_data = [{"keyword": kw, "avg_cpc": round(1.23 * (i + 1), 2), "search_volume": 1000 + i * 100} for i, kw in enumerate(keyword_list)]
            
            # Visualize the data
            plot_url = create_keyword_visualizations(keyword_data)
            
            # Render the results in the template
            return render_template('keyword_results.html', keyword_data=keyword_data, plot_url=plot_url)
        except Exception as e:
            return f"Error processing keyword planner data: {str(e)}", 500

    return render_template('keyword_planner.html')

def create_keyword_visualizations(keyword_data):
    keywords = [data['keyword'] for data in keyword_data]
    avg_cpc = [data['avg_cpc'] for data in keyword_data]
    search_volume = [data['search_volume'] for data in keyword_data]
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Bar chart for Average CPC
    ax[0].bar(keywords, avg_cpc, color='#ff5733')
    ax[0].set_ylabel('Average CPC')
    ax[0].set_title('Average CPC per Keyword')
    ax[0].tick_params(axis='x', rotation=45)

    # Bar chart for Search Volume
    ax[1].bar(keywords, search_volume, color='#33c4ff')
    ax[1].set_ylabel('Search Volume')
    ax[1].set_title('Search Volume per Keyword')
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save and encode image
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url

if __name__ == '__main__':
    # Ensure the app is running within an application context
    with app.app_context():
        db.create_all()  # This will create the tables if they don't exist
    app.run(debug=True)