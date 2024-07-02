# Movie Recommendation System

This repository contains the implementation of a movie recommendation system using collaborative filtering and content-based filtering techniques. The system utilizes data from the MovieLens dataset to provide movie recommendations based on user ratings and movie genres.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Explanation](#code-explanation)
6. [Results](#results)
7. [Contributors](#contributors)
8. [License](#license)

## Project Overview

The goal of this project is to build a recommendation system that suggests movies to users based on their past ratings and movie content. The project involves:

- Data loading and preprocessing
- Exploratory data analysis
- Collaborative filtering using matrix factorization
- Content-based filtering using movie genres
- Combining collaborative and content-based recommendations
- Visualizing results

## Dataset

The project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) which contains 100,000 ratings from 943 users on 1682 movies. The dataset includes:

- User ratings (u.data)
- Movie information (u.item)
- User information (u.user)
- Genre information (u.genre)
- Occupation information (u.occupation)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ElectroBuzz/Recommendation-System.git
    cd Recommendation-System
    ```

2. Install the required libraries:
    ```sh
    pip install numpy pandas matplotlib seaborn scikit-learn scipy
    ```

3. Download the MovieLens dataset and place the files in the `data` directory.

## Usage

1. Run the script:
    ```sh
    python Recommendation_System.py
    ```

2. Follow the prompts to enter your user ID and the last movie you watched to get movie recommendations.

## Code Explanation

### Data Loading and Preprocessing

The data is loaded from various files and preprocessed for further analysis.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data files
u_ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
u_ratings = pd.read_csv('u.data', sep='\t', names=u_ratings_columns)

u_movies_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
u_movies = pd.read_csv('u.item', sep='|', names=u_movies_columns, encoding='latin-1')

u_user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
u_user = pd.read_csv('u.user', sep='|', names=u_user_columns)

u_occupation_columns = ['occupation']
u_occupation = pd.read_csv('u.occupation', names=u_occupation_columns)
```

### Exploratory Data Analysis

Performing exploratory data analysis to understand the data distribution and patterns.

```python
# Plot the percentage of users by occupation
occupation_counts = u_user['occupation'].value_counts()
occupation_percentages = (occupation_counts / u_user.shape[0]) * 100

plt.figure(figsize=(8, 6))
occupation_percentages.plot(kind='bar', color='skyblue')
plt.title('Percentage of Users by Occupation')
plt.xlabel('Occupation')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Collaborative Filtering

Using Non-negative Matrix Factorization (NMF) for collaborative filtering to predict user ratings for movies.

```python
from sklearn.decomposition import NMF

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(u_ratings)

model = NMF(n_components=20, init='random', random_state=0, max_iter=500)
W = model.fit_transform(X)
H = model.components_
Q = np.dot(W, H)
```

### Content-Based Filtering

Using movie genres to recommend similar movies based on content.

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(movie_genres, movie_genres)

def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if u_movies_new['title'].iloc[x[0]] != title]
    sim_scores = sim_scores[:n_recommendations]
    similar_movies = [i[0] for i in sim_scores]
    print(f"Because you watched {title}:")
    print(u_movies_new['title'].iloc[similar_movies].tolist())
    return u_movies_new['movie_id'].iloc[similar_movies].tolist()
```

### Combining Recommendations

Combining collaborative and content-based recommendations to provide a final list of recommended movies.

```python
last_watched_movie = movie_finder(get_last_watched_movie())
content_list_movies = get_content_based_recommendations(last_watched_movie, 20)
similar_users_demo = find_similar_users(user_idf, merged1_df_encoded)

mean_ratings_list = calculate_mean_ratings(Q, content_list_movies, similar_users_demo, movie_mapper, user_mapper)
sorted_movie_ratings = dict(sorted(movie_ratings_dict.items(), key=lambda item: item[1], reverse=True))
```

## Results

The recommendation system provides a list of movie recommendations based on user ratings and movie content. It combines collaborative filtering and content-based filtering techniques to generate personalized recommendations.

## Contributors

- Raman Verma [GitHub Profile](https://github.com/ElectroBuzz)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
