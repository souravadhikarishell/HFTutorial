from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

#performs summarization
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

#performs refinement
refinement_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

#perform question answering
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

#summary template
summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}")

#LangChain implementation of summarization chain to pass execution between different models
summarization_chain = summary_template | summarizer | refiner

text_to_summarize = input("\nEnter text to summarize:\n")
length = input("\nEnter the length (short/medium/long): ")

#Invoke the LangChain
summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

#Print Output/Results
print("\n🔹 **Generated Summary:**")
print(summary)

while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
    if question.lower() == "exit":
        break

    qa_result = qa_pipeline(question=question, context=summary)

    print("\n🔹 **Answer:**")
    print(qa_result["answer"])