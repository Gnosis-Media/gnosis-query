import requests
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configuration
SEARCH_SERVICE_URL = "http://localhost:5000" # 'http://34.207.126.237:80'
# EMBEDDING_SERVICE_URL = 'http://44.203.255.78:80'

def test_search_flow():
    """Test the complete search flow"""
    
    # Test cases
    test_cases = [
        {
            'user_id': 51,  # Replace with a real user ID from your database
            'query': """
tes and grant unemployment compensation only create unemployment.
It is absurd to demand that European wages must be raised because wages are higher in the U.S. than in Europe. If the immigration barriers to the U.S., Australia, et cetera, would be removed, European workers could emigrate, which would gradually lead to an international equalization of wage rates.
The permanent unemployment of hundreds of thousands and millions of people on the one hand, and the consump- tion of capital on the other hand, are each consequences of interventionism's artificial raising of wage rates by labor unions and unemployment compensation.
5. Destruction Resulting from Intervention
The history of the last decades can be understood only with a comprehension of the consequences of such inter- vention in the economic operations of the private property
29""

order. Since the demise of classical liberalism, intervention- ism has been the gist of politics in all countries in Europe and America.
The economic layman only observes that "interested par- ties" succeed again and again in escaping the strictures of law. The fact that the system functions poorly is blamed ex- clusively on the law that does not go far enough, and on cor- ruption that prevents its application. The very failure of interventionism reinforces the layman's conviction that pri- vate property must be controlled severely. The corruption of the regulatory bodies does not shake his blind confidence in the infallibility and perfection',
""",
            'limit': 3
        },
        {
            'user_id': 7,  # Replace with another user ID
            'query': 'how does mises explain the interventionism governments in global affairs',
            'limit': 5
        }
    ]
    
    for test_case in test_cases:
        logging.info(f"\nTesting search with parameters:")
        logging.info(f"User ID: {test_case['user_id']}")
        logging.info(f"Query: {test_case['query']}")
        logging.info(f"Limit: {test_case['limit']}")
        
        try:
            # Make the search request
            response = requests.get(
                f"{SEARCH_SERVICE_URL}/api/search",
                params={
                    'user_id': test_case['user_id'],
                    'query': test_case['query'],
                    'limit': test_case['limit']
                },
                headers={
                    'X-API-KEY': 'the-most-super-secret-key'
                }
            )
            
            # Check response
            if response.status_code == 200:
                results = response.json()
                logging.info("\nSearch Results:")
                logging.info(f"Number of results: {len(results['results'])}")
                
                # Print each result with details
                for idx, result in enumerate(results['results'], 1):
                    logging.info(f"\nResult {idx}:")
                    logging.info(f"Chunk ID: {result['chunk_id']}")
                    logging.info(f"Content ID: {result['content_id']}")
                    logging.info(f"File Name: {result['file_name']}")
                    logging.info(f"Similarity Score: {result['similarity_score']:.4f}")
                    logging.info(f"Preview: {result['text'][:200]}...")
            else:
                logging.error(f"Search failed with status code: {response.status_code}")
                logging.error(f"Error message: {response.json()}")
                
        except Exception as e:
            logging.error(f"Test failed with error: {str(e)}")

# def test_direct_embedding_similarity():
#     """Test the embedding service directly"""
    
#     test_cases = [
#         {
#             'text': 'machine learning and artificial intelligence',
#             'embedding_ids': [1, 2, 3, 4, 5],  # Replace with real embedding IDs
#             'limit': 3
#         }
#     ]
    
#     for test_case in test_cases:
#         logging.info(f"\nTesting embedding similarity with parameters:")
#         logging.info(f"Query: {test_case['text']}")
#         logging.info(f"Embedding IDs: {test_case['embedding_ids']}")
        
#         try:
#             response = requests.post(
#                 f"{EMBEDDING_SERVICE_URL}/api/embedding/similar",
#                 json=test_case,
#                 headers={
#                     'X-API-KEY': 'the-most-super-secret-key'
#                 }
#             )
            
#             if response.status_code == 200:
#                 results = response.json()
#                 logging.info("\nSimilar Embeddings:")
#                 pprint(results)
#             else:
#                 logging.error(f"Similarity search failed with status code: {response.status_code}")
#                 logging.error(f"Error message: {response.json()}")
                
#         except Exception as e:
#             logging.error(f"Test failed with error: {str(e)}")

def run_tests():
    """Run all tests"""
    logging.info("Starting tests...")
    
    logging.info("\n=== Testing Complete Search Flow ===")
    test_search_flow()
    
    logging.info("\n=== Testing Direct Embedding Similarity ===")
    # test_direct_embedding_similarity()
    
    logging.info("\nTests completed!")

if __name__ == "__main__":
    run_tests()