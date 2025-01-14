from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from openai import OpenAI

def load_amazon_data(file_path, sample_size=None):
    """Load Amazon reviews data from TSV file"""
    try:
        if sample_size:
            df = pd.read_csv(file_path, 
                           sep='\t',
                           nrows=sample_size)
        else:
            df = pd.read_csv(file_path, sep='\t')
        
        # Basic data processing
        df['review_date'] = pd.to_datetime(df['review_date'])
        
        # Print basic information
        print("\nDataset Information:")
        print(f"Total reviews: {len(df)}")
        print(f"Time span: {df['review_date'].min()} to {df['review_date'].max()}")
        print(f"\nProduct categories:\n{df['product_category'].value_counts()}")
        print(f"\nRating distribution:\n{df['star_rating'].value_counts().sort_index()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

class CustomerSupportAgent:
    def __init__(self, api_key, reviews_data_path, sample_size=None):
        """Initialize the Customer Support Agent."""
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        self.tools = {
            "query_handler": self.handle_product_query,
            "review_analyzer": self.analyze_reviews,
            "satisfaction_analyzer": self.analyze_satisfaction,
            "similar_cases": self.find_similar_cases,
            "generate_response": self.generate_support_response
        }
        
        # Load the Amazon reviews data
        print("Loading Amazon reviews data...")
        self.reviews_df = load_amazon_data(reviews_data_path, sample_size)
        if self.reviews_df is None:
            raise ValueError("Failed to load reviews data")
            
        self.chat_history = []
        self.logs = []
        self.client = OpenAI(api_key=api_key)
        
        print("Customer Support Agent initialized successfully!")

    def get_llm_decision(self, query):
        """Analyze customer query and decide on appropriate support action."""
        system_prompt = """You are a customer support AI assistant.
        For each query, you should:
        1. Identify the most appropriate support tool
        2. Generate a plan to help the customer
        
        Available support tools:
        - query_handler: For product-specific questions and information
        - review_analyzer: For analyzing product reviews and feedback
        - satisfaction_analyzer: For analyzing customer satisfaction patterns
        - similar_cases: For finding similar customer cases and solutions
        - generate_response: For generating appropriate support responses
        
        Respond in JSON format:
        {
            "tool": "selected_tool_name",
            "reasoning": "explanation of why this tool was chosen",
            "execution_plan": "step by step support plan",
            "priority": "high/medium/low"
        }"""

        try:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Customer query: {query}"}
                ],
                temperature=0.2
            )
            
            response = completion.choices[0].message.content
            return json.loads(response)
            
        except Exception as e:
            self.logs.append(f"[{datetime.now()}] Error in LLM decision: {str(e)}")
            raise

    def handle_product_query(self, query, execution_plan):
        """Handle product-specific queries using review data."""
        try:
            prompt = """Based on the customer query, what product information should we look for?
            Return in JSON format:
            {
                "product_category": "category if mentioned",
                "key_features": ["list of features to look for"],
                "concern_type": "type of customer concern"
            }"""
            
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            query_analysis = json.loads(completion.choices[0].message.content)
            relevant_reviews = self.reviews_df[
                self.reviews_df['product_category'] == query_analysis['product_category']
            ] if query_analysis['product_category'] else self.reviews_df
            
            summary = self._summarize_product_info(relevant_reviews, query_analysis['key_features'])
            return self.format_product_info(summary)
            
        except Exception as e:
            return f"Error handling product query: {str(e)}"

    def analyze_reviews(self, query, execution_plan):
        """Analyze product reviews with improved formatting."""
        try:
            prompt = """What aspects of the reviews should we analyze?
            Return in JSON format:
            {
                "aspects": ["list of aspects to analyze"],
                "sentiment": "positive/negative/both",
                "time_range": "recent/all"
            }"""
            
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            analysis_params = json.loads(completion.choices[0].message.content)
            relevant_reviews = self._filter_reviews(analysis_params)
            
            analysis = {
                'common_issues': self._extract_common_issues(relevant_reviews),
                'review_count': len(relevant_reviews)
            }
            
            return self.format_review_analysis(analysis)
            
        except Exception as e:
            return f"Error analyzing reviews: {str(e)}"

    def analyze_satisfaction(self, query, execution_plan):
        """Analyze customer satisfaction with improved formatting."""
        try:
            reviews_subset = self.reviews_df.copy()
            
            metrics = {
                'average_rating': reviews_subset['star_rating'].mean(),
                'rating_distribution': reviews_subset['star_rating'].value_counts(),
                'verified_purchase_satisfaction': reviews_subset[
                    reviews_subset['verified_purchase'] == 'Y'
                ]['star_rating'].mean(),
                'recent_trend': self._calculate_satisfaction_trend(reviews_subset)
            }
            
            return self.format_satisfaction_response(metrics)
            
        except Exception as e:
            return f"Error analyzing satisfaction: {str(e)}"

    def find_similar_cases(self, query, execution_plan):
        """Find and format similar customer cases."""
        try:
            prompt = """Extract key aspects of this customer query.
            Focus on:
            - Main issue/concern
            - Product type if mentioned
            - Specific problems described"""
            
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            similar_reviews = self._find_similar_reviews(completion.choices[0].message.content)
            return self.format_similar_cases(similar_reviews)
            
        except Exception as e:
            return f"Error finding similar cases: {str(e)}"

    def generate_support_response(self, query, execution_plan):
        """Generate formatted customer support response."""
        try:
            similar_cases = self._find_similar_reviews(query)
            
            prompt = f"""Generate a helpful customer support response.
            Context:
            - Customer Query: {query}
            - Similar Cases: {similar_cases}
            
            Response should be:
            - Professional and empathetic
            - Specific to the customer's concern
            - Include relevant information from similar cases
            - Provide clear next steps if needed"""
            
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ]
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def format_satisfaction_response(self, metrics):
        """Format satisfaction metrics into readable response."""
        response = (
            f"Based on {metrics['rating_distribution'].sum()} reviews:\n\n"
            f"• Overall rating: {metrics['average_rating']:.1f} out of 5 stars\n"
            f"• Rating breakdown:\n"
        )
        
        total_reviews = sum(metrics['rating_distribution'].values())
        for rating in sorted(metrics['rating_distribution'].index, reverse=True):
            count = metrics['rating_distribution'][rating]
            percentage = (count / total_reviews) * 100
            response += f"  - {rating} stars: {count} reviews ({percentage:.1f}%)\n"
        
        recent_months = sorted(metrics['recent_trend'].items(), key=lambda x: x[0])[-3:]
        response += "\nRecent trends:\n"
        for period, rating in recent_months:
            response += f"  - {period}: {rating:.1f} stars\n"
            
        return response

    def format_review_analysis(self, analysis):
        """Format review analysis into readable response."""
        response = f"I've analyzed {analysis['review_count']} relevant reviews.\n\n"
        
        if analysis['common_issues']['common_headlines']:
            response += "Common issues reported:\n"
            for headline, count in analysis['common_issues']['common_headlines'].items():
                response += f"• {headline} (mentioned {count} times)\n"
            
            response += "\nDetailed examples:\n"
            for i, review in enumerate(analysis['common_issues']['sample_reviews'], 1):
                review_excerpt = review[:200] + '...' if len(review) > 200 else review
                response += f"\n{i}. {review_excerpt}\n"
                
        return response

    def format_similar_cases(self, similar_reviews):
        """Format similar reviews into readable response."""
        if not similar_reviews:
            return "I couldn't find any similar cases in our reviews."
            
        response = "I found some similar experiences from other customers:\n\n"
        
        for i, review in enumerate(similar_reviews, 1):
            response += (
                f"{i}. {review['review_headline']}\n"
                f"Rating: {'⭐' * int(review['star_rating'])}\n"
                f"{review['review_body'][:200]}...\n\n"
            )
            
        return response

    def format_product_info(self, summary):
        """Format product information into readable response."""
        response = (
            f"Based on {summary['total_reviews']} reviews:\n\n"
            f"• Average Rating: {summary['average_rating']:.1f} stars\n\n"
            "Key Points:\n"
        )
        
        for point in summary['key_points']:
            response += f"• {point}\n"
            
        return response

    def _summarize_product_info(self, reviews, key_features):
        """Summarize product information."""
        summary = {
            'total_reviews': len(reviews),
            'average_rating': reviews['star_rating'].mean(),
            'key_points': []
        }
        
        # Extract key points from reviews
        for feature in key_features:
            relevant_reviews = reviews[
                reviews['review_body'].str.contains(feature, case=False, na=False)
            ]
            if not relevant_reviews.empty:
                avg_rating = relevant_reviews['star_rating'].mean()
                summary['key_points'].append(
                    f"{feature}: {avg_rating:.1f} stars from {len(relevant_reviews)} mentions"
                )
                
        return summary

    def _filter_reviews(self, params):
        """Filter reviews based on parameters."""
        filtered_df = self.reviews_df.copy()
        
        if params['sentiment'] == 'positive':
            filtered_df = filtered_df[filtered_df['star_rating'] >= 4]
        elif params['sentiment'] == 'negative':
            filtered_df = filtered_df[filtered_df['star_rating'] <= 2]
            
        if params['time_range'] == 'recent':
            # Get last 3 months of reviews
            latest_date = filtered_df['review_date'].max()
            three_months_ago = latest_date - pd.DateOffset(months=3)
            filtered_df = filtered_df[filtered_df['review_date'] >= three_months_ago]
            
        return filtered_df

    def _extract_common_issues(self, reviews):
        """Extract common issues from negative reviews."""
        negative_reviews = reviews[reviews['star_rating'] <= 2]
        
        common_issues = (
            negative_reviews['review_headline']
            .value_counts()
            .head(5)
            .to_dict()
        )
        
        sample_reviews = negative_reviews['review_body'].head(3).tolist()
        
        return {
            'common_headlines': common_issues,
            'sample_reviews': sample_reviews
        }

    def _calculate_satisfaction_trend(self, reviews):
        """Calculate satisfaction trends."""
        reviews['review_date'] = pd.to_datetime(reviews['review_date'])
        monthly_satisfaction = reviews.groupby(
            reviews['review_date'].dt.to_period('M')
        )['star_rating'].mean()
        return monthly_satisfaction.to_dict()

    def _find_similar_reviews(self, query_aspects):
        """Find reviews similar to the query."""
        keywords = query_aspects.lower().split()
        
        relevant_reviews = self.reviews_df[
            self.reviews_df['review_body'].str.lower().str.contains(
                '|'.join(keywords),
                na=False
            )
        ]
        
        similar_reviews = (
            relevant_reviews
            .sort_values('helpful_votes', ascending=False)
            .head(3)
        )
        
        return similar_reviews[['review_headline', 'review_body', 'star_rating']].to_dict('records')

    def chat_loop(self):
        """Interactive chat loop for customer support."""
        print("Customer Support Bot: Hello! How can I help you today? (Type 'exit' to end)")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Customer Support Bot: Thank you for chatting! Have a great day!")
                break
                
            try:
                decision = self.get_llm_decision(user_input)
                if decision["tool"] in self.tools:
                    response = self.tools[decision["tool"]](user_input, decision["execution_plan"])
                else:
                    response = "I apologize, but I'm not sure how to help with that specific request."
                    
                print(f"Customer Support Bot: {response}")
                
            except Exception as e:
                print(f"Customer Support Bot: I apologize, but I encountered an error: {str(e)}")

if __name__ == "__main__":
    # Initialize agent with sample size for testing
    try:
        agent = CustomerSupportAgent(
            api_key="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",  # Replace with your OpenAI API key
            reviews_data_path="amazon.tsv",  # Path to your Amazon reviews file
            sample_size=10000  # Load 10k reviews for testing
        )
        
        # Test queries to demonstrate different tools
        print("\nTesting Individual Tools:")
        
        print("\n1. Product Query Test:")
        query = "What do customers say about the battery life?"
        response = agent.handle_product_query(query, None)
        print(f"Query: {query}")
        print(f"Response: {response}")
        
        print("\n2. Review Analysis Test:")
        query = "What are the most common complaints?"
        response = agent.analyze_reviews(query, None)
        print(f"Query: {query}")
        print(f"Response: {response}")
        
        print("\n3. Satisfaction Analysis Test:")
        query = "How satisfied are customers overall?"
        response = agent.analyze_satisfaction(query, None)
        print(f"Query: {query}")
        print(f"Response: {response}")
        
        print("\n4. Similar Cases Test:")
        query = "My device won't turn on"
        response = agent.find_similar_cases(query, None)
        print(f"Query: {query}")
        print(f"Response: {response}")
        
        print("\n5. Support Response Test:")
        query = "How do I reset my device?"
        response = agent.generate_support_response(query, None)
        print(f"Query: {query}")
        print(f"Response: {response}")
        
        print("\nStarting Interactive Chat Mode:")
        agent.chat_loop()
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        
    finally:
        print("\nExecution completed.")