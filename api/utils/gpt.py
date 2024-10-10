from openai import OpenAI
from llama_index import StorageContext, load_index_from_storage
import os


def read_from_storage(persist_dir):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)

def get_template():

    query = """
    [BEGIN CONTEXT]
        STNWeb is a new web tool for the visualization of the behavior of optimization algorithms such as metaheuristics. It allows for the graphical analysis of multiple runs of multiple algorithms on the same problem instance and, in this way, it facilitates the understanding of algorithm behavior. It may help, for example, in identifying the reasons for a rather low algorithm performance. This, in turn, can help the algorithm designer to change the algorithm in order to improve its performance. STNWeb is designed to be user-friendly. Moreover, it is offered for free to the research community.
    [END CONTEXT]

    [BEGIN TASK A]
        [BEGIN RULES]
            These are the general rules of the system:
            (1) The more nodes pointing to the best fitness (this doesn't assume that it represents the global optimum), the higher the algorithm's quality because it can find the best result more reliably.
            (2) The algorithm that has more overlap (merge) is likely to be more robust. If the algorithm finds nodes with the best fitness.
            (3) For a minimization problem, indicating that an algorithm is superior involves favoring a smaller average fitness value. Whereas in the case of maximization, declaring an algorithm as better requires a higher average fitness value.
        [END RULES]
        [BEGIN DATA]
            Problem:
                This is a {{type_problem}} problem.
            Features:
                {{features}}
            [END DATA]
        [END TASK A]

        [BEGIN QUERIES]
            Task A: Identify the most effective algorithm for the considered optimization problem and provide detailed insights.
            Instructions for your response:
            - Clearly specify the winning algorithm by enclosing its name within brackets; in case of a tie, denote [draw].
            - Present your response between [BEGIN TASK A] and [END TASK A].
        [END QUERIES]
    """
    return query

def get_response(openapi_key, query):
    
    client = OpenAI(
        api_key= openapi_key
    )
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": query}],
        temperature=0.7,
        )

    return response.choices[0].message.content

