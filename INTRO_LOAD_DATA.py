import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery
import requests
import json

# OPENAI_API_KEY = 'sk-proj-phXYdnnbnFH2Bv7eILVtT3BlbkFJjz1TVJVUjc6b1adTJg0N'
OPENAI_API_KEY = 'sk-proj-3adqfe46QqtPHNPgdIUFT3BlbkFJvpuRdh9t9KOWSbs3OqBJ'
COHERE_API_KEY = "rSRj9TyZ5DMUxdhatPhmBGrXrBzv53UjPClbY0In"


def demoCohereEmbedding():
    # Connect with Weaviate Embedded
    client = weaviate.connect_to_embedded(
        version="1.23.7",  # run Weaviate 1.23.7
        headers={
            "X-OpenAI-Api-Key": OPENAI_API_KEY,
            "X-OpenAI-organization": "org - 5l4DJClhbz6RrhGJ1j817eFY",
            "X-Cohere-Api-Key": COHERE_API_KEY
        })

    if client.is_ready():
        print(f"\n---> Weaviate Client is connected <---\n")

    # Create a collection here - with Cohere as a vectorizer
    schemaName = "Questions"
    if client.collections.exists(schemaName):
        client.collections.delete(schemaName)

    client.collections.create(
        name=schemaName,
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_cohere())

    def load_data(path):
        resp = requests.get(path)
        return json.loads(resp.text)

    data_10 = load_data(
        "https://raw.githubusercontent.com/weaviate-tutorials/multimodal-workshop/main/1-intro/jeopardy_tiny.json")

    print(f"\n---> My Question Bank.... <---\n")

    print(json.dumps(data_10, indent=2))

    # Insert Many
    questions = client.collections.get(schemaName)
    questions.data.insert_many(data_10)

    # Show preview w/o vectors
    '''questions = client.collections.get("Questions")
    response = questions.query.fetch_objects(limit=2)
    for item in response.objects:
        print(item.uuid, item.properties)'''

    # Show preview with vector
    questions = client.collections.get("Questions")
    response = questions.query.fetch_objects(
        limit=4,
        include_vector=True
    )

    for item in response.objects:
        print(item.properties)
        print(item.vector, '\n')

    # Super quick query example
    query = "whaat is weather"
    print(f"\n---> Response on query: {query} .... <---\n")
    response = questions.query.near_text(
        query,
        # "Zwierzęta afrykańskie", #African animals in Polish
        # "アフリカの動物", #African animals in Japanese
        limit=2,
        certainty=0.7,
        return_metadata=MetadataQuery(distance=True)
    )

    for item in response.objects:
        print(item.properties, flush=True)
        print(f"Distance: {item.metadata.distance}")
        print(f"Confidence: {1.0 - item.metadata.distance / 2.0}")

    # Wrap up
    client.close()


def demoOPENAIEmbedding():
    # Connect with Weaviate Embedded
    client = weaviate.connect_to_embedded(
        version="1.23.7",  # run Weaviate 1.23.7
        headers={
            "X-OpenAI-Api-Key": OPENAI_API_KEY,
            "X-Cohere-Api-Key": COHERE_API_KEY
        })

    if client.is_ready():
        print(f"Weaviate Client is connected")

    # Create a collection here - with Cohere as a vectorizer
    schemaName = "Questions"
    if client.collections.exists(schemaName):
        client.collections.delete(schemaName)

    client.collections.create(
        name=schemaName,
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        generative_config=wvc.config.Configure.Generative.openai(model="gpt-4"))

    def load_data(path):
        resp = requests.get(path)
        return json.loads(resp.text)

    data_1000 = load_data(
        "https://raw.githubusercontent.com/weaviate-tutorials/multimodal-workshop/main/1-intro/jeopardy_1k.json")

    # print(json.dumps(data_1000, indent=2))

    # Insert Many
    questions = client.collections.get(schemaName)
    questions.data.insert_many(data_1000[:100])

    # RAG example
    questions = client.collections.get("Questions")
    response = questions.query.fetch_objects(limit=2)
    for item in response.objects:
        print(item.uuid, item.properties)

    # Show preview with vector
    questions = client.collections.get("Questions")
    response = questions.query.fetch_objects(
        limit=2,
        include_vector=True
    )

    for item in response.objects:
        print(item.properties)
        print(item.vector, '\n')

    # Super quick query example
    response = questions.query.near_text(
        "dinasour",
        # "Zwierzęta afrykańskie", #African animals in Polish
        # "アフリカの動物", #African animals in Japanese
        limit=2
    )

    for item in response.objects:
        print(item.properties, flush=True)

    # Wrap up
    client.close()


if __name__ == "__main__":
    demoCohereEmbedding()
    # demoOPENAIEmbedding()
