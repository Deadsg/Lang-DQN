import numpy as np
import gym
import random
import argparse
from typing import Tuple, Type
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, StructuredTool, BaseTool
import requests
from langchain.pydantic_v1 import BaseModel, Field

class train_dqn_agent(BaseModel):
    a: int = Field(description="Agent layer")
    b: int = Field(description="Dqn result")

# Define the Deep Q-Network class
class DQN:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = {}
        model['W1'] = np.random.randn(self.state_size, 24) / np.sqrt(self.state_size)
        model['b1'] = np.zeros((1, 24))
        model['W2'] = np.random.randn(24, 24) / np.sqrt(24)
        model['b2'] = np.zeros((1, 24))
        model['W3'] = np.random.randn(24, self.action_size) / np.sqrt(24)
        model['b3'] = np.zeros((1, self.action_size))
        return model

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.predict(state)
        return np.argmax(q_values)

    def predict(self, state: np.ndarray) -> np.ndarray:
        hidden_layer1 = np.dot(state, self.model['W1']) + self.model['b1']
        hidden_layer1 = np.tanh(hidden_layer1)
        hidden_layer2 = np.dot(hidden_layer1, self.model['W2']) + self.model['b2']
        hidden_layer2 = np.tanh(hidden_layer2)
        q_values = np.dot(hidden_layer2, self.model['W3']) + self.model['b3']
        return q_values

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                q_values_next = self.predict(next_state)
                target = reward + self.gamma * np.amax(q_values_next)
            q_values = self.predict(state)
            q_values[0][action] = target
            self.fit(state, q_values)

    def fit(self, state: np.ndarray, q_values: np.ndarray):
        hidden_layer1 = np.dot(state, self.model['W1']) + self.model['b1']
        hidden_layer1 = np.tanh(hidden_layer1)
        hidden_layer2 = np.dot(hidden_layer1, self.model['W2']) + self.model['b2']
        hidden_layer2 = np.tanh(hidden_layer2)
        output_layer = np.dot(hidden_layer2, self.model['W3']) + self.model['b3']
        loss = np.mean(np.square(q_values - output_layer))
        d_output = 2 * (output_layer - q_values) / q_values.shape[0]
        d_hidden2 = np.dot(d_output, self.model['W3'].T) * (1 - np.power(hidden_layer2, 2))
        d_hidden1 = np.dot(d_hidden2, self.model['W2'].T) * (1 - np.power(hidden_layer1, 2))
        dW3 = np.dot(hidden_layer2.T, d_output)
        db3 = np.sum(d_output, axis=0, keepdims=True)
        dW2 = np.dot(hidden_layer1.T, d_hidden2)
        db2 = np.sum(d_hidden2, axis=0)
        dW1 = np.dot(state.T, d_hidden1)
        db1 = np.sum(d_hidden1, axis=0)
        self.model['W3'] -= self.learning_rate * dW3
        self.model['b3'] -= self.learning_rate * db3
        self.model['W2'] -= self.learning_rate * dW2
        self.model['b2'] -= self.learning_rate * db2
        self.model['W1'] -= self.learning_rate * dW1
        self.model['b1'] -= self.learning_rate * db1

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create the environment
class Environment:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

class custom_dqn_agent(BaseTool):
    name = "DQN Agent"
    description = "Logical Reasoning for Langchain Integration."
    args_schema: Type[BaseModel] = train_dqn_agent
    return_direct: bool = True

    def _run(
        self, DQN: int, chat: int
    ) -> str:
        """Use the tool."""
        return DQN * chat

# Define the custom LangChain tool for training and testing the DQN agent
@tool("train_dqn_agent", return_direct=True, args_schema=train_dqn_agent)
def train_dqn_agent(num_episodes: int, batch_size: int) -> str:
    """Train and test a DQN agent on the CartPole-v1 environment."""
    env = Environment()
    agent = DQN(env.state_size, env.action_size)
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(batch_size)
        agent.update_epsilon()

        print(f"Episode: {episode+1}, Reward: {total_reward}")

    # Test the trained agent
    test_episodes = 10
    test_rewards = []

    for _ in range(test_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            state = next_state
            total_reward += reward

        test_rewards.append(total_reward)

    average_reward = np.mean(test_rewards)
    env.close()

    return f"Training completed. Average reward over {test_episodes} test episodes: {average_reward}"

def chat():
    parser = argparse.ArgumentParser(description="CLI Chatbot")
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes for DQN training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DQN training")
    args = parser.parse_args()

    while True:
        user_input = input("You: ")  # Get user input from command line
        if user_input.lower() == "terminate":
            break

        try:
            # Initialize output parser
            output_parser = StrOutputParser()

            # Initialize LLM model
            llm = Ollama(model=args.model)

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("user", user_input)
            ])

            # Create tool manager and add the tool
            tool_chain = StructuredTool.from_function(
                name="train_dqn_agent",
                description="Train and test a DQN agent on the CartPole-v1 environment",
                func=train_dqn_agent,
                args_schema=train_dqn_agent,  # Correct usage here
            )

            # Create chain
            chain = (prompt | tool_chain | output_parser | llm)

            # Invoke the chain and get the response
            response = chain.invoke({
                "input": user_input,
                "num_episodes": args.num_episodes,
                "batch_size": args.batch_size
            })

            print(f"DQN Results: {tool_chain.invoke}")
            print(f"Model Response: {response}")

        except requests.exceptions.ConnectionError:
            print("Error: Unable to connect to the local Ollama service. Please ensure it is running.")

if __name__ == "__main__":
    chat()
