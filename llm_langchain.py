from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from langchain_experimental.utilities.python import PythonREPL
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate

class LangChainToolAgent:
    def __init__(self, api_key):
        """Initialize the LangChain-based Tool Agent."""
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        self.logs = []
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            api_key=api_key  # Updated to use api_key instead of openai_api_key
        )
        
        # Initialize tools
        self.setup_tools()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def setup_tools(self):
        """Setup LangChain tools."""
        # Calculator tool using LangChain's math chain
        llm_math_chain = LLMMathChain.from_llm(llm=self.llm, verbose=True)
        
        # Python REPL for code execution
        python_repl = PythonREPLTool()
        
        # Define custom tools
        self.tools = [
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="Useful for performing mathematical calculations. Input should be a mathematical expression."
            ),
            Tool(
                name="Plotter",
                func=self.use_plotter,
                description="Creates visualizations and plots. Input should describe the type of plot needed."
            ),
            Tool(
                name="CodeExecutor",
                func=self.use_code_executor,
                description="Generates and executes Python code. Input should describe the code functionality needed."
            ),
            Tool(
                name="APITool",
                func=self.use_api_caller,
                description="Makes API calls to external services. Input should describe the API request needed."
            ),
            python_repl  # Built-in Python REPL tool for safe code execution
        ]

    def use_plotter(self, query):
        """Enhanced plotting tool with LangChain safety features."""
        try:
            self.logs.append(f"[{datetime.now()}] Using plotter tool for query: {query}")
            
            # Use LangChain's Python REPL tool for safe code execution
            python_repl = PythonREPLTool()
            
            # Generate plotting code using LLM
            code_prompt = PromptTemplate(
                input_variables=["query"],
                template="""
                Write Python code to create this plot using matplotlib:
                Query: {query}
                
                Important:
                - Use 'plt.figure()' to create a new figure
                - Include proper labels and title
                - Don't include any explanations, just the code
                """
            )
            
            code = self.llm.predict(code_prompt.format(query=query))
            self.logs.append(f"[{datetime.now()}] Generated plotting code: {code}")
            
            # Add necessary imports and execute
            setup_code = "import matplotlib.pyplot as plt\nimport numpy as np\n"
            result = python_repl.run(setup_code + code)
            plt.show()
            
            return "Plot generated successfully."
            
        except Exception as e:
            self.logs.append(f"[{datetime.now()}] Error in plotting: {str(e)}")
            return f"Error in plotting: {str(e)}"

    def use_code_executor(self, query):
        """Safe code execution using LangChain's Python REPL."""
        try:
            self.logs.append(f"[{datetime.now()}] Using code executor for query: {query}")
            
            python_repl = PythonREPLTool()
            
            # Generate code using LLM
            code_prompt = PromptTemplate(
                input_variables=["query"],
                template="Write Python code to: {query}\nInclude only the code, no explanations."
            )
            
            code = self.llm.predict(code_prompt.format(query=query))
            self.logs.append(f"[{datetime.now()}] Generated code: {code}")
            
            # Execute code safely
            result = python_repl.run(code)
            return f"Code execution result: {result}"
            
        except Exception as e:
            self.logs.append(f"[{datetime.now()}] Error in code execution: {str(e)}")
            return f"Error in code execution: {str(e)}"

    def use_api_caller(self, query):
        """Enhanced API caller using LangChain's safety features."""
        try:
            self.logs.append(f"[{datetime.now()}] Using API caller for query: {query}")
            
            # For weather queries, use Open-Meteo API
            if "weather" in query.lower():
                response = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": 51.5074,  # London coordinates
                        "longitude": -0.1278,
                        "current_weather": True
                    }
                )
                return f"API Response: {response.json()}"
            
            return "API endpoint not configured for this query type."
            
        except Exception as e:
            self.logs.append(f"[{datetime.now()}] Error in API call: {str(e)}")
            return f"Error in API call: {str(e)}"

    def decide_and_execute(self, query):
        """Use LangChain agent to process query and execute appropriate tools."""
        try:
            self.logs.append(f"[{datetime.now()}] Processing query: {query}")
            response = self.agent.run(query)
            return response
        except Exception as e:
            self.logs.append(f"[{datetime.now()}] Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def inspect_steps(self):
        """Return the logged steps for inspection."""
        return "\n".join(self.logs)

    def evaluate(self, test_cases):
        """Evaluate the system on test cases."""
        results = []
        for i, (query, expected) in enumerate(test_cases):
            self.logs.append(f"[{datetime.now()}] Evaluating test case {i+1}: {query}")
            response = self.decide_and_execute(query)
            results.append({
                "query": query,
                "response": response,
                "expected": expected,
                "success": response == expected if expected is not None else True
            })
        return results

# Example usage
if __name__ == "__main__":
    # Initialize agent with your OpenAI API key
    agent = LangChainToolAgent("")
    
    # Test cases
    test_cases = [
        ("Calculate 2 + 2", "4"),
        ("Calculate the square root of 16 plus 5", "9"),
        ("Plot a sine wave with amplitude 2", "Plot generated successfully."),
        ("Get the current weather in London", None)  # Response will vary
    ]
    
    # Interactive test
    print("Interactive Session:")
    query = "Calculate 2 + 2"
    response = agent.decide_and_execute(query)
    print(f"Query: {query}")
    print(f"Response: {response}")
    
    # Evaluate test cases
    print("\nEvaluating Test Cases:")
    results = agent.evaluate(test_cases)
    for result in results:
        print(f"\nQuery: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"Success: {result['success']}")
    
    # Show execution logs
    print("\nExecution Logs:")
    print(agent.inspect_steps())