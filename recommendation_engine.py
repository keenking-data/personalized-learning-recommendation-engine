import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
import logging
import pickle
from datetime import datetime
from typing import List, Dict, Tuple
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    raise

class CourseRecommender:
    def __init__(self, data_paths: List[str], focus_subjects: List[str] = ['Web Development', 'Data Science']):
        """
        Initialize the recommender with dataset paths and focus subjects.
        
        Args:
            data_paths: List of paths to CSV datasets.
            focus_subjects: List of subjects to prioritize.
        """
        self.data_paths = data_paths
        self.focus_subjects = focus_subjects
        self.data = None
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.tfidf_matrix = None
        self.model_file = 'tfidf_vectorizer.pkl'

    def load_and_combine_data(self) -> None:
        """
        Load and combine datasets into a unified schema.
        """
        try:
            dfs = []
            for path in self.data_paths:
                df = pd.read_csv(path)
                if 'EdX' in path:
                    df = self._process_edx(df)
                elif 'udemy' in path.lower():
                    df = self._process_udemy(df)
                else:
                    logger.warning(f"Unknown dataset format for {path}. Skipping.")
                    continue
                dfs.append(df)
            self.data = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {len(self.data)} courses from {len(dfs)} datasets.")
        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            raise

    def _process_edx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process EdX dataset to match unified schema.
        """
        try:
            return pd.DataFrame({
                'course_id': [str(uuid.uuid4()) for _ in range(len(df))],
                'title': df['Name'],
                'description': df['Course Description'].fillna(''),
                'subject': df['Course Description'].apply(self._infer_subject),
                'difficulty': df['Difficulty Level'],
                'price': 0.0,
                'is_paid': False,
                'duration': np.nan,
                'institution': df['University'],
                'url': df['Link'],
                'num_subscribers': np.nan,
                'num_reviews': np.nan,
                'num_lectures': np.nan,
                'published_year': np.nan,
                'quality_score': np.nan
            })
        except Exception as e:
            logger.error(f"Error processing EdX dataset: {e}")
            raise

    def _process_udemy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Udemy dataset to match unified schema.
        """
        try:
            return pd.DataFrame({
                'course_id': df['course_id'].astype(str),
                'title': df['course_title'],
                'description': df['course_title'],  # Udemy lacks detailed description
                'subject': df['subject'],
                'difficulty': df['level'],
                'price': df['price'],
                'is_paid': df['is_paid'],
                'duration': df['content_duration'],
                'institution': 'Udemy',
                'url': df['url'],
                'num_subscribers': df['num_subscribers'],
                'num_reviews': df['num_reviews'],
                'num_lectures': df['num_lectures'],
                'published_year': pd.to_datetime(df['published_timestamp']).dt.year,
                'quality_score': df['num_reviews'] / df['num_subscribers'].replace(0, 1)
            })
        except Exception as e:
            logger.error(f"Error processing Udemy dataset: {e}")
            raise

    def _infer_subject(self, description: str) -> str:
        """
        Infer subject from description for EdX courses.
        """
        description = description.lower()
        for subject in self.focus_subjects:
            if subject.lower() in description:
                return subject
        return 'Other'

    def clean_data(self) -> None:
        """
        Clean the combined dataset.
        """
        try:
            # Remove duplicates
            self.data = self.data.drop_duplicates(subset=['title', 'institution'])
            logger.info(f"Removed duplicates. Remaining courses: {len(self.data)}")

            # Handle missing values
            self.data['price'].fillna(0, inplace=True)
            self.data['difficulty'].fillna('All Levels', inplace=True)
            self.data['description'].fillna('', inplace=True)
            self.data['num_subscribers'].fillna(0, inplace=True)
            self.data['num_reviews'].fillna(0, inplace=True)
            self.data['num_lectures'].fillna(self.data['num_lectures'].median(), inplace=True)
            self.data['duration'].fillna(self.data['duration'].median(), inplace=True)
            self.data['published_year'].fillna(self.data['published_year'].median(), inplace=True)

            # Cap outliers
            duration_cap = self.data['duration'].quantile(0.95)
            lectures_cap = self.data['num_lectures'].quantile(0.95)
            self.data = self.data[
                (self.data['duration'] <= duration_cap) & 
                (self.data['num_lectures'] <= lectures_cap)
            ]

            # Filter relevant subjects
            self.data = self.data[self.data['subject'].isin(self.focus_subjects)]
            logger.info(f"Filtered to {len(self.data)} courses in {self.focus_subjects}")
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            raise

    def engineer_features(self) -> None:
        """
        Engineer features for recommendation.
        """
        try:
            # Encode difficulty
            difficulty_mapping = {
                'Beginner Level': 0, 'Intermediate Level': 1, 'Advanced Level': 2, 
                'All Levels': 0.5, 'Beginner': 0, 'Intermediate': 1, 'Advanced': 2
            }
            self.data['difficulty_encoded'] = self.data['difficulty'].map(difficulty_mapping).fillna(0.5)

            # Derive recency
            self.data['recency'] = 2025 - self.data['published_year']

            # Derive intensity
            self.data['intensity'] = self.data['num_lectures'] / self.data['duration'].replace(0, 1)

            # Price tiers
            self.data['price_tier'] = pd.cut(
                self.data['price'], 
                bins=[-1, 0, 50, float('inf')], 
                labels=['Free', 'Low-Cost', 'High-Cost']
            )

            # TF-IDF for title and description
            self.data['text'] = self.data['title'] + ' ' + self.data['description']
            self.data['text'] = self.data['text'].apply(
                lambda x: ' '.join([t for t in word_tokenize(x.lower()) if t.isalnum() and t not in stopwords.words('english')])
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.data['text'])
            logger.info("Features engineered: difficulty_encoded, recency, intensity, price_tier, TF-IDF")

            # Save vectorizer for production
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise

    def recommend(self, user_profile: Dict, top_n: int = 5) -> pd.DataFrame:
        """
        Generate recommendations based on user profile.
        
        Args:
            user_profile: Dict with keys: interests (List[str]), max_price (float), 
                         difficulty (str), max_duration (float).
            top_n: Number of recommendations to return.
        
        Returns:
            DataFrame with recommended courses.
        """
        try:
            # Knowledge-based filtering
            filtered_data = self.data.copy()
            if user_profile.get('max_price'):
                filtered_data = filtered_data[filtered_data['price'] <= user_profile['max_price']]
            if user_profile.get('difficulty'):
                filtered_data = filtered_data[filtered_data['difficulty'].str.contains(user_profile['difficulty'], case=False)]
            if user_profile.get('max_duration'):
                filtered_data = filtered_data[filtered_data['duration'] <= user_profile['max_duration']]
            if user_profile.get('subjects'):
                filtered_data = filtered_data[filtered_data['subject'].isin(user_profile['subjects'])]

            if filtered_data.empty:
                logger.warning("No courses match user constraints.")
                return pd.DataFrame()

            # Content-based filtering
            user_text = ' '.join(user_profile.get('interests', []))
            user_vector = self.vectorizer.transform([user_text])
            similarities = cosine_similarity(user_vector, self.tfidf_matrix[filtered_data.index]).flatten()
            filtered_data['similarity'] = similarities

            # Combine with quality and recency
            filtered_data['score'] = (
                0.6 * filtered_data['similarity'] + 
                0.2 * filtered_data['quality_score'].fillna(0) + 
                0.2 * filtered_data['recency'] / filtered_data['recency'].max()
            )

            # Return top N recommendations
            recommendations = filtered_data.sort_values('score', ascending=False)[
                ['course_id', 'title', 'subject', 'difficulty', 'price', 'url', 'score']
            ].head(top_n)
            logger.info(f"Generated {len(recommendations)} recommendations.")
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return pd.DataFrame()

    def evaluate(self, test_profiles: List[Dict], ground_truth: Dict[str, List[str]]) -> Dict:
        """
        Evaluate the model using precision, recall, F1-score, and NDCG.
        
        Args:
            test_profiles: List of user profiles for testing.
            ground_truth: Dict mapping user_id to list of relevant course_ids.
        
        Returns:
            Dict with evaluation metrics.
        """
        try:
            y_true, y_pred = [], []
            ndcg_scores = []
            for profile in test_profiles:
                user_id = profile.get('user_id')
                if not user_id or user_id not in ground_truth:
                    continue
                recommendations = self.recommend(profile, top_n=5)
                predicted_ids = recommendations['course_id'].tolist()
                true_ids = ground_truth.get(user_id, [])
                
                # Binary relevance (1 if in ground truth, 0 otherwise)
                y_true.extend([1 if course_id in true_ids else 0 for course_id in predicted_ids])
                y_pred.extend([1] * len(predicted_ids))  # Assume all recommended are predicted relevant
                
                # NDCG
                relevance = [1 if course_id in true_ids else 0 for course_id in predicted_ids]
                if relevance:
                    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevance))
                    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_ids), len(relevance))))
                    ndcg = dcg / idcg if idcg > 0 else 0
                    ndcg_scores.append(ndcg)

            metrics = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0
            }
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}

def main():
    # Example usage
    try:
        recommender = CourseRecommender(
            data_paths=[
                'EdX.csv',
                'udemy_online_education_courses_dataset.csv'
            ],
            focus_subjects=['Web Development', 'Data Science']
        )
        
        # Load and process data
        recommender.load_and_combine_data()
        recommender.clean_data()
        recommender.engineer_features()

        # Example user profile
        user_profile = {
            'interests': ['Python', 'Web Development', 'JavaScript'],
            'max_price': 50.0,
            'difficulty': 'Beginner',
            'max_duration': 5.0,
            'subjects': ['Web Development']
        }

        # Generate recommendations
        recommendations = recommender.recommend(user_profile)
        print("Recommendations:\n", recommendations)

        # Evaluate with sample ground truth
        test_profiles = [user_profile]
        ground_truth = {'user1': recommendations['course_id'].tolist()[:2]}  # Simulated ground truth
        user_profile['user_id'] = 'user1'
        metrics = recommender.evaluate(test_profiles, ground_truth)
        print("Evaluation Metrics:\n", metrics)

        # Save processed data
        recommender.data.to_csv('processed_courses.csv', index=False)
        logger.info("Saved processed dataset to 'processed_courses.csv'")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()