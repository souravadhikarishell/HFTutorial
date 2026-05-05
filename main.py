from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# model = pipeline("summarization", model="facebook/bart-large-cnn")
# response = model("text to summarize")
# print(response)

model= pipeline(task="text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.1",
                max_length=256,
                truncation=True,
                )

llm = HuggingFacePipeline(pipeline=model)

# Create the prompt template
template = PromptTemplate.from_template("Explain {topic} in detail for a {age} year old to understand.")

chain = template | llm
topic = input("Topic: ")
age = input("Age: ")

#Execute the chain
response = chain.invoke({"topic": topic, "age": age})
print(response)