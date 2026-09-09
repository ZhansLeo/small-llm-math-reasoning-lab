from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()


model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b


@tool
def get_temperature(city: str) -> str:
    """获取指定城市当前气温"""
    data = {
        "北京": "32°C",
        "东京": "27°C",
        "南京": "30°C",
    }
    return data.get(city, "未知城市")


agent = create_agent(
    model=model,
    tools=[add, multiply, get_temperature],
)


for chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "先计算 123 + 456，再把结果乘以 10。"
            }
        ]
    },
    stream_mode="updates",
):
    print(chunk)