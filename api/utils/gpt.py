from openai import OpenAI
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, download_loader, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI
import os

client = OpenAI(
    api_key= os.environ.get("OPENAI_API_KEY")
)

def read_from_storage(persist_dir):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)

def get_template():

    query = ("Task A: \n"
            "Now, considering these rules, take a look at this data: {info_algorithm}\n"
            "----------------\n"
            "Task B: \n"
            "These are the parameters by agglomerative clustering algorithm: {agglomerative_params} \n"
            "----------------\n"
            "These are the configuration constant: \n"
            "\t minimum possible number of cluster: {min_cluster} \n"
            "\t maximum possible number of cluster: {max_cluster} \n"
            "----------------\n"
            "Resolving Tasks: \n"
            "\t Number 1 (Task A): Give me a general interpretation that allows comparing both algorithms and determining which one is better. \n"
            "\t Number 2 (Task B): Give me the new numerical values for the agglomerative clustering to improve the results of the visualization in STNWeb, and explain why. These parameters work for all algorithms. \n" 
            "\t Number 3: Explanation of the results Task A and B in the context of the article Search trajectory Networks Web (STNWeb). \n"
            "----------------\n"
                    "Rules to response: \n"
                    "\t Your answer should be in the opening and closing tag for each Task. For example <Task_A> ... </Task_A> \n"
                    "\t You response must has a limit to 300 tokens with details. \n"
                    "\t In the answer add bold the name of each algorithm. \n"
                    "\t This is a {type_problem} optimization problem. \n")
    return query

def get_response(query):
    index = read_from_storage("/home/camilocs/research/stnweb/analytics/storage")

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.2, model_name="gpt-4-1106-preview"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    custom_llm_query_engine = index.as_query_engine(service_context=service_context)

    return str(custom_llm_query_engine.query(query))

