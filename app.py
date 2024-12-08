from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import requests
import logging
import os
from datetime import datetime
from secrets_manager import get_service_secrets

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

app = Flask(__name__)
CORS(app)

secrets = get_service_secrets('gnosis-query')

C_PORT = int(secrets.get('PORT', 5000))
SQLALCHEMY_DATABASE_URI = (
    f"mysql+pymysql://{secrets['MYSQL_USER']}:{secrets['MYSQL_PASSWORD_CONTENT']}"
    f"@{secrets['MYSQL_HOST']}:{secrets['MYSQL_PORT']}/{secrets['MYSQL_DATABASE']}"
)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

API_KEY = secrets.get('API_KEY')
EMBEDDING_API_URL = secrets.get('EMBEDDING_API_URL')

db = SQLAlchemy(app)

# Define database models (same as content processor)
class Content(db.Model):
    __tablename__ = 'content'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    file_size = db.Column(db.Integer, nullable=False)
    s3_key = db.Column(db.String(255))
    chunk_count = db.Column(db.Integer, default=0)
    custom_prompt = db.Column(db.Text)
    # metadata
    title = db.Column(db.String(255))
    author = db.Column(db.String(255))
    publication_date = db.Column(db.Date)
    publisher = db.Column(db.String(255))
    source_language = db.Column(db.String(255))
    genre = db.Column(db.String(255))
    topic = db.Column(db.Text)

class ContentChunk(db.Model):
    __tablename__ = 'content_chunk'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    chunk_order = db.Column(db.Integer, nullable=False)
    chunk_text = db.Column(db.Text, nullable=False)
    embedding_id = db.Column(db.Integer)

@app.route('/api/search', methods=['GET'])
def search_similar_chunks():
    """
    Search for similar chunks across a user's content
    Required query parameters:
    - user_id: ID of the user
    - query: Text to search for
    Optional query parameters:
    - content_id: ID of specific content to search within
    - limit: Number of results to return (default: 5)
    """
    # Get and validate parameters
    user_id = request.args.get('user_id')
    query_text = request.args.get('query')
    content_id = request.args.get('content_id', type=int)
    limit = request.args.get('limit', default=5, type=int)

    if not user_id or not query_text:
        return jsonify({
            'error': 'Missing required parameters: user_id and query'
        }), 400

    try:
        # Step 1: Get content IDs to search within
        content_query = db.session.query(Content.id)\
            .filter(Content.user_id == user_id)
        
        if content_id:
            # If content_id provided, only search within that content
            content_query = content_query.filter(Content.id == content_id)
            
        content_ids = content_query.all()
        content_ids = [id[0] for id in content_ids]

        if not content_ids:
            return jsonify({
                'message': 'No content found for this user',
                'results': []
            }), 200

        # Step 2: Get all embedding IDs for the user's content
        embedding_chunks = db.session.query(ContentChunk.id, ContentChunk.embedding_id)\
            .filter(ContentChunk.content_id.in_(content_ids))\
            .filter(ContentChunk.embedding_id.isnot(None))\
            .all()

        if not embedding_chunks:
            return jsonify({
                'message': 'No embeddings found for this user\'s content',
                'results': []
            }), 200

        # Create mapping of embedding_id to chunk_id
        embedding_to_chunk = {ec[1]: ec[0] for ec in embedding_chunks}
        embedding_ids = list(embedding_to_chunk.keys())

        # Step 3: Query embedding service for similar embeddings
        headers = {'X-API-KEY': API_KEY}
        response = requests.post(
            f'{EMBEDDING_API_URL}/api/embedding/similar',
            headers=headers,
            json={
                'text': query_text,
                'embedding_ids': embedding_ids,
                'limit': limit
            }
        )

        if response.status_code != 200:
            return jsonify({
                'error': 'Failed to get similar embeddings'
            }), 500

        similar_embeddings = response.json()['similar_embeddings']

        # Step 4: Get the corresponding chunks
        similar_chunk_ids = [
            embedding_to_chunk[emb['id']] 
            for emb in similar_embeddings
        ]

        # Step 5: Get the chunk details
        similar_chunks = db.session.query(
            ContentChunk.id,
            ContentChunk.chunk_text,
            ContentChunk.content_id,
            Content.file_name
        )\
            .join(Content)\
            .filter(ContentChunk.id.in_(similar_chunk_ids))\
            .all()

        # Format results
        results = [{
            'chunk_id': chunk.id,
            'content_id': chunk.content_id,
            'file_name': chunk.file_name,
            'text': chunk.chunk_text,
            'similarity_score': next(
                emb['similarity_score'] 
                for emb in similar_embeddings 
                if embedding_to_chunk[emb['id']] == chunk.id
            )
        } for chunk in similar_chunks]

        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return jsonify({
            'message': 'Search completed successfully',
            'results': results
        }), 200

    except Exception as e:
        logging.error(f"Error in search_similar_chunks: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500
    
# Get content by ID
@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_content_by_id(content_id):
    try:
        content = Content.query.get(content_id)
        if content is None:
            return jsonify({'error': 'Content not found'}), 404
        return jsonify({
            'id': content.id,
            'user_id': content.user_id,
            'file_name': content.file_name,
            'file_type': content.file_type,
            'upload_date': content.upload_date.isoformat(),
            'file_size': content.file_size,
            's3_key': content.s3_key,
            'chunk_count': content.chunk_count,
            'custom_prompt': content.custom_prompt,
            'title': content.title,
            'author': content.author,
            'publication_date': content.publication_date.isoformat() if content.publication_date else None,
            'publisher': content.publisher,
            'source_language': content.source_language,
            'genre': content.genre,
            'topic': content.topic
        }), 200
    except Exception as e:
        logging.error(f"Error in get_content_by_id: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

# Get text and embedding_id of a chunk by ID
@app.route('/api/chunk/<int:chunk_id>', methods=['GET'])
def get_chunk_text_by_id(chunk_id):
    try:
        chunk = ContentChunk.query.get(chunk_id)
        if chunk is None:
            return jsonify({'error': 'Chunk not found'}), 404
        return jsonify({
            'text': chunk.chunk_text,
            'embedding_id': chunk.embedding_id
        }), 200
    except Exception as e:
        logging.error(f"Error in get_chunk_text_by_id: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# add middleware
@app.before_request
def log_request_info():
    logging.info(f"Headers: {request.headers}")
    logging.info(f"Body: {request.get_data()}")

    # for now just check that it has a Authorization header
    if 'X-API-KEY' not in request.headers:
        logging.warning("No X-API-KEY header")
        return jsonify({'error': 'No X-API-KEY'}), 401
    
    x_api_key = request.headers.get('X-API-KEY')
    if x_api_key != API_KEY:
        logging.warning("Invalid X-API-KEY")
        return jsonify({'error': 'Invalid X-API-KEY'}), 401
    else:
        return

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=C_PORT)