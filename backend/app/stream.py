import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4.1-mini")

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser

async def streaming():
    async for event in chain.stream({"topic": "parrot"}):
        # if kind == "on_chat_model_stream":
        print(event, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(streaming())