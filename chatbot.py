from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import time

class ChatBot:
    def __init__(self, vector_db, chosen_model):
        self.db = vector_db
        match chosen_model:
            case "Deepseek":
                self.llm = ChatOllama(
                    model = "deepseek-r1:1.5b",
                    base_url = "http://localhost:11434",
                    temperature = 0.3
                )
            case "Qwen":
                self.llm = ChatOllama(
                    model = "qwen2.5:1.5b",
                    base_url = "http://localhost:11434",
                    temperature = 0.3
                )
            case "gemma3":
                self.llm = ChatOllama(
                    model = "gemma3:1b",
                    base_url = "http://localhost:11434",
                    temperature = 0.3
                )
            case "Llama":
                self.llm = ChatOllama(
                    model = "llama3.2:3b",
                    base_url = "http://localhost:11434",
                    temperature = 0.3
                )
        self.starttime = 0
        self.endtime = 0
        
        self.prompt_template = """
            You are an AI assistant tasked with answering questions based solely
            on the provided context. Your goal is to generate a comprehensive answer
            for the given question using only the information available in the context.

            context: {context}
            question: {question}

            <response> Your answer in Markdown format. </response>
        """

        self.chain = self.build_chain()
    
    def build_chain(self):
        prompt = PromptTemplate(template = self.prompt_template, 
                                input_variables = ["context", "question"])
        retriever = self.db.as_retriever(search_kwargs = {"k": 5})

        chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff",
            retriever = retriever,
            return_source_documents = True,
            chain_type_kwargs = {"prompt": prompt},
            verbose = True
        )

        return chain

    def qa(self, question):
        self.starttime = time.time()
        response = self.chain(question)
        return response["result"]
        self.endtime = time.time()
    