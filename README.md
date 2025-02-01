# 🤖 Customer Support LLM Agent

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-API-orange)
![License](https://img.shields.io/badge/license-MIT-green)

An intelligent customer support agent powered by Large Language Models that provides advanced product analysis, sentiment tracking, and automated response generation for e-commerce platforms.

## ✨ Features

- 🔍 **Dynamic Product Search**: Advanced semantic search capabilities for finding specific products and similar items
- 📊 **Statistical Analysis**: Comprehensive product statistics and performance metrics
- 💭 **Sentiment Analysis**: In-depth analysis of customer reviews with sentiment tracking over time
- 📈 **Visual Analytics**: Rich visualizations including word clouds, sentiment trends, and rating distributions
- 🤝 **Smart Response Generation**: Context-aware customer support response generation
- 🔄 **Tool Selection**: Automatic selection of appropriate analysis tools based on query context
- 📝 **Execution Tracking**: Detailed tracking and visualization of query execution steps

## 🏗️ Architecture

The agent is built with a modular architecture consisting of:

- **DynamicTool**: A flexible code generation and execution engine
- **CustomerSupportAgent**: Main agent class handling query processing and tool orchestration
- **Analysis Modules**: Specialized modules for sentiment analysis, product statistics, and response generation

### Technology Stack

- Python 3.8+
- OpenAI GPT-4 API
- Pandas & NumPy for data processing
- Matplotlib & Seaborn for visualization
- NLTK & TextBlob for NLP tasks
- scikit-learn for text vectorization and similarity matching

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-support-llm-agent.git
cd customer-support-llm-agent
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```python
export OPENAI_API_KEY='your-api-key-here'
```

## 📖 Usage

### Basic Usage

```python
from customer_support_agent import CustomerSupportAgent

# Initialize the agent
agent = CustomerSupportAgent(
    api_key='your-openai-api-key',
    dataset_path='path/to/your/data.tsv'
)

# Process a query
result = agent.process_query("Analyze customer ratings for wireless headphones")

# Visualize execution
agent.visualize_execution(result)
```

### Example Queries

```python
# Analyze product sentiment
result = agent.process_query("Show sentiment analysis for Bluetooth speakers")

# Get product statistics
result = agent.process_query("Get statistics for gaming headsets")

# Compare products
result = agent.process_query("Compare reviews between different brands of wireless earbuds")
```

## 📊 Output Examples

The agent provides rich visual output including:

- Word clouds of customer reviews
- Sentiment trend analysis over time
- Rating distribution charts
- Execution timeline visualizations
- Customer satisfaction metrics

## 🛠️ Configuration

The agent can be configured through various parameters:

```python
agent = CustomerSupportAgent(
    api_key='your-api-key',
    dataset_path='data.tsv',
    model="gpt-4",  # OpenAI model to use
    temperature=0.7  # Response creativity level
)
```

## 📝 Data Format

The agent expects a TSV file with the following columns:
- marketplace
- customer_id
- review_id
- product_id
- product_parent
- product_title
- product_category
- star_rating
- helpful_votes
- total_votes
- vine
- verified_purchase
- review_headline
- review_body
- review_date

## 🔄 Performance Optimization

For optimal performance:
- Use efficient data loading with appropriate chunk sizes
- Implement caching for frequently accessed data
- Utilize batch processing for multiple queries
- Enable parallel processing for independent operations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT-4 API
- The open-source community for various dependencies
- Contributors and maintainers

## 📞 Support

For support and questions, please open an issue in the GitHub repository or contact the maintainers directly.

---
Made with ❤️ by [Your Name/Organization]
