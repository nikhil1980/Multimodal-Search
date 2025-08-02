import weaviate
import weaviate.classes as wvc
import json
from embedding_util import generate_embeddings

from weaviate.classes.query import MetadataQuery

import warnings

warnings.filterwarnings("ignore")

# Uncomment and replace with your key
#COHERE_API_KEY = XXX  # Replace with your Cohere API key
#HF_TOKEN = XXX  # Replace with your HuggingFace token


jioData = [
    {
        "Category": "PERVER",
        "Question": "He is the only engineer who works on all technologies",
        "Answer": "Kimanshu Khatia"
    },
    {
        "Category": "PERVER",
        "Question": "These are key expertise needed in to work in Perver team",
        "Answer": "Java, Python, Cpluplus, Sql"
    },
    {
        "Category": "PERVER",
        "Question": "The coolest person in the server team",
        "Answer": "Prashish Zishra"
    },
    {
        "Category": "QA",
        "Question": "Well versed in all forms of testing",
        "Answer": "Daruth"
    },
    {
        "Category": "QA",
        "Question": "She is the main lead for Server QA",
        "Answer": "Aadhu"
    },
    {
        "Category": "PRODUCT",
        "Question": "A tool that is used by Product team to create wireframes and design",
        "Answer": "Vigma"
    },
    {
        "Category": "PRODATABASE",
        "Question": "Core tool used by database team for their work",
        "Answer": "SQLDeveloper"
    },
    {
        "Category": "PRODATABASE",
        "Question": "Most proficient database engineer",
        "Answer": "Wrabhat"
    },
    {
        "Category": "CLIENT",
        "Question": "Engineer well versed in android",
        "Answer": "Diddesh"
    },
    {
        "Category": "CLIENT",
        "Question": "Client where there is always an issue of badlight view",
        "Answer": "Android"
    }
]


def demo1():
    client = weaviate.Client(
        url="http://localhost:8080",  # Replace with your endpoint
    )

    # Just to illustrate a simple Weaviate health check, if part of a larger system
    print('Weviate DB is ready.') if client.is_ready() else print('Weviate DB is down.')

    # Class definition object. Weaviate's autoschema feature will infer properties
    # when importing.
    class_obj = {
        "class": "DocumentSearch",
        "vectorizer": "none",
    }

    if client.schema.exists(class_name="DocumentSearch"):
        client.schema.delete_class(class_name="DocumentSearch")
        # Add the class to the schema
        client.schema.create_class(class_obj)

    # test source documents
    documents = [
        "A group of vibrant parrots chatter loudly, sharing stories of their tropical adventures.",
        "The mathematician found solace in numbers, deciphering the hidden patterns of the universe.",
        "The robot, with its intricate circuitry and precise movements, assembles the devices swiftly.",
        "The chef, with a sprinkle of spices and a dash of love, creates culinary masterpieces.",
        "The ancient tree, with its gnarled branches and deep roots, whispers secrets of the past.",
        "The detective, with keen observation and logical reasoning, unravels the intricate web of clues.",
        "The sunset paints the sky with shades of orange, pink, and purple, reflecting on the calm sea.",
        "In the dense forest, the howl of a lone wolf echoes, blending with the symphony of the night.",
        "The dancer, with graceful moves and expressive gestures, tells a story without uttering a word.",
        "In the quantum realm, particles flicker in and out of existence, dancing to the tunes of probability."]

    # Configure a batch process. Since our "documents" is small, just setting the
    # whole batch to the size of the "documents" list
    client.batch.configure(batch_size=len(documents))
    with client.batch as batch:
        for i, doc in enumerate(documents):
            print(f"document: {i + 1} added to DB")

            properties = {
                "source_text": doc,
            }
            vector = generate_embeddings(doc)

            batch.add_data_object(properties, "DocumentSearch", vector=vector)

    # test query
    query = "Give me some content about the birds"
    query_vector = generate_embeddings(query)

    # The default metric for ranking documents is by cosine distance.
    # Cosine Similarity = 1 - Cosine Distance
    result = client.query.get(
        "DocumentSearch", ["source_text"]
    ).with_near_vector(
        {
            "vector": query_vector,
            "certainty": 0.7
        }
    ).with_limit(2).with_additional(['certainty', 'distance']).do()

    print(f"\nQuery: {query}")
    print(f"\nQuery vector dim: {len(query_vector)} and content: {query_vector}")
    print(f"\n\nResponse...")
    print(json.dumps(result, indent=4))

    # delete schema

    client.schema.delete_all()


def demo2():
    client = weaviate.connect_to_local(headers={
        "X-Cohere-Api-Key": COHERE_API_KEY,
        "X-HuggingFace-Api-Key": HF_TOKEN,
    }, skip_init_checks=True)

    # Just to illustrate a simple Weaviate health check, if part of a larger system
    print('Weviate DB is ready.', flush=True) if client.is_ready() else print('Weviate DB is down.', flush=True)

    meta_info = client.get_meta()
    print(meta_info)

    # Create a collection here - with own vectorizer
    schemaName = "PcloudDemo"
    if client.collections.exists(schemaName):
        client.collections.delete(schemaName)

    my_collection = client.collections.create(
        name=schemaName,
        description="Basic Semantic Search demo for PCloud over Weviate",
        vectorizer_config=[
            wvc.config.Configure.NamedVectors.text2vec_cohere(name="category_emb"),
            wvc.config.Configure.NamedVectors.text2vec_cohere(name="question_emb"),
            wvc.config.Configure.NamedVectors.text2vec_cohere(name="answer_emb"),

        ],
        properties=[  # Define properties
            wvc.config.Property(name="answer", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="question", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
        ],
    )

    #print(my_collection)

    question_objs = list()
    for i, d in enumerate(jioData):
        question_objs.append(
            {
                "answer": d["Answer"],
                "question": d["Question"],
                "category": d["Category"],
            },
        )
    '''vector={
        "answer_emb": generate_embeddings(d["Answer"]),
        "question_emb": generate_embeddings(d["Question"]),
        "category_emb": generate_embeddings(d["Category"])
    }'''

    questions = client.collections.get(schemaName)
    with questions.batch.dynamic() as batch:
        for q in question_objs:
            batch.add_object(properties=q)

    #questions.data.insert_many(question_objs)  # This uses batching under the hood

    questions = client.collections.get(schemaName)
    response = questions.query.fetch_objects(include_vector=True, limit=2, offset=1)

    for o in response.objects:
        print(o.properties)
        print(len(o.vector['answer_emb']), o.vector['answer_emb'][:3])

    questions = client.collections.get(schemaName)

    query = "Give me some programming languges used by Pcloud"
    query = '''मुझे Pcloud द्वारा उपयोग की जाने वाली कुछ प्रोग्रामिंग भाषाएँ दीजिए'''
    response = questions.query.near_text(
        query=query,
        include_vector="True",
        target_vector="question_emb",
        certainty=0.7,
        limit=4,
        return_metadata=MetadataQuery(distance=True)
    )

    print(f"\nQuery: {query}")
    print(f"\nResponse...")

    for r in response.objects:
        print(r.properties)
        print(f"Distance: {r.metadata.distance}")
        print(f"Confidence: {1.0 - r.metadata.distance / 2.0}")

    client.close()


if __name__ == '__main__':
    print(f"\n---> Demo 1 for semantic search on single text modality using my custom embedding <---\n")
    demo1()
    print(f"\n---> Demo 2 for semantic search on multiple text streams using my COHERE embedding <---\n")
    demo2()
