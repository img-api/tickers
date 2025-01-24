import os
import os.path
import json

from custom_components import ExtractFoodItemsFromImage

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
# from haystack.components.generators.openai import OpenAIGenerator
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator
from haystack.components.builders import PromptBuilder
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorKeywordRetriever
from haystack.components.builders.answer_builder import AnswerBuilder


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/dev/vertex-ai-config.json"


# Load environment variables
load_dotenv()


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_PATH = os.environ.get("BASE_PATH", "../../data")
TICKERS_PATH = os.path.join(BASE_PATH, "tickers")
PHOTOS_PATH = os.path.join(BASE_PATH, "example_food_photos")


class Ticker:
    def __init__(self, news=None, recreate_table=False):
        self.news = news

        # Initialize PostgreSQL Document Store
        print("INITIALIZING DOCUMENT STORE")

        self.document_store = PgvectorDocumentStore(
            embedding_dimension=768,
            vector_function="cosine_similarity",
            recreate_table=recreate_table,
            search_strategy="hnsw",
        )
        print("DOCUMENT STORE INITIALIZED")

    def get_tickers(self, ingredients: str):
        template = """
        Given the following information, answer the question.

        Context: 

        {% for document in documents %}
            :
            {{ document.content }}

        {% endfor %}

        Question: Could you extract all the stock tickers from the this article?

        {{ news }}

        Add the instructions for that recipe, formatted as markdown.
        """
        pipe = Pipeline()

        retriever = PgvectorKeywordRetriever(document_store=self.document_store)
        pipe.add_component("retriever", retriever)
        pipe.add_component(instance=PromptBuilder(template=template), name="prompt_builder")
        pipe.add_component(instance=VertexAIGeminiGenerator(), name="llm")
        pipe.add_component(instance=AnswerBuilder(), name="answer_builder")
        pipe.connect("retriever", "prompt_builder.documents")
        pipe.connect("prompt_builder", "llm")
        pipe.connect("llm.replies", "answer_builder.replies")
        # pipe.connect("llm.meta", "answer_builder.meta")
        pipe.connect("retriever", "answer_builder.documents")

        result = pipe.run(
            {
                "retriever": {"query": self.news},
                "prompt_builder": {"news": self.news},
                "answer_builder": {"query": self.news},
            }
        )
        tickers = result["answer_builder"]["answers"][0].data
        print(tickers)
        return tickers

    def insert_documents(self, contents):
        print("INSERTING DOCUMENTS")

        for content in contents:
            doc = Document(content=content, embedding=self.generate_doc_embedding(content))
            print("DOCUMENT EMBEDDING GENERATED")

            try:
                self.document_store.write_documents([doc])
                print("DOCUMENT INSERTED")
            except Exception as e:
                print(e)

        print("DOCUMENTS INSERTED")

    def generate_doc_embedding(self, text):
        doc = Document(content="text")
        doc_embedder = SentenceTransformersDocumentEmbedder()
        doc_embedder.warm_up()

        result = doc_embedder.run([doc])
        return result['documents'][0].embedding

    def get_image_ingredients(self, image_number: int):
        image_path = os.path.join(PHOTOS_PATH, f"food{image_number}.webp")
        print(image_path)

        pipe = Pipeline()
        pipe.add_component(instance=ExtractFoodItemsFromImage(), name="image_extractor")

        result = pipe.run(
            {
                "image_extractor": {"image_path": image_path}
            }
        )
        print(result)
        return result["image_extractor"]["answer"]



def load_tickers():
    print("loading TICKERS")

    tickers = []

    for f in os.listdir(TICKERS_PATH):
        full_path = os.path.join(TICKERS_PATH, f)
        if not os.path.isfile(full_path):
            continue

        with open(full_path) as f:
            # read the file as json
            companies = json.load(f)

            for company in companies["companies"]:
                try:
                    company_name = company["company_name"]
                except KeyError as e:
                    print(e)
                    continue

                exchange_tickers = ",".join(company["exchange_tickers"])
                related_exchange_tickers =  ",".join(company.get("related_exchange_tickers", []))

                tickers.append(
                    f"""
                    Company name: {company_name}
                    Exchange tickers: {exchange_tickers},
                    Related exchange tickers: {related_exchange_tickers}
                    """
                )

    print(f"{len(tickers)} TICKERS loaded")

    return tickers


def load_news():
    print("loading NEWS")

    news = []

    with open(os.path.join(BASE_PATH, "ai_summary.json")) as f:
        # read the file as json
        articles = json.load(f)

        for article in articles["news"]:
            news.append(
                """
                Title: {title}
                Summary: {summary}
                """.format(
                    title=article["title"],
                    summary=article["summary"],
                )
            )

    print(f"{len(news)} NEWS loaded")

    return news


if __name__ == '__main__':
    tickers = load_tickers()
    ticker_extractor = Ticker(recreate_table=True)
    ticker_extractor.insert_documents(tickers)
