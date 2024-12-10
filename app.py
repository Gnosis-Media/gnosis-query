from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_restx import Api, Resource, fields
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
app.config['DEBUG'] = True

# Configure API
api = Api(app,
    version='1.0',
    title='Gnosis Query API',
    description='API for searching and retrieving content chunks',
    doc='/docs'
)

# Configure namespace
ns = api.namespace('api', description='Query operations')

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

# API Models
search_response = api.model('SearchResponse', {
    'message': fields.String,
    'results': fields.List(fields.Nested(api.model('SearchResult', {
        'chunk_id': fields.Integer,
        'content_id': fields.Integer,
        'file_name': fields.String,
        'text': fields.String,
        'similarity_score': fields.Float
    })))
})

content_response = api.model('ContentResponse', {
    'id': fields.Integer,
    'user_id': fields.Integer,
    'file_name': fields.String,
    'file_type': fields.String,
    'upload_date': fields.String,
    'file_size': fields.Integer,
    's3_key': fields.String,
    'chunk_count': fields.Integer,
    'custom_prompt': fields.String,
    'title': fields.String,
    'author': fields.String,
    'publication_date': fields.String,
    'publisher': fields.String,
    'source_language': fields.String,
    'genre': fields.String,
    'topic': fields.String
})

chunk_response = api.model('ChunkResponse', {
    'text': fields.String,
    'embedding_id': fields.Integer
})

@ns.route('/search')
class SearchResource(Resource):
    @api.doc('search_similar_chunks',
             params={'user_id': 'ID of the user',
                    'query': 'Text to search for',
                    'content_id': 'ID of specific content to search within',
                    'limit': 'Number of results to return (default: 5)'})
    @api.marshal_with(search_response)
    def get(self):
        """Search for similar chunks across a user's content"""
        user_id = request.args.get('user_id')
        query_text = request.args.get('query')
        content_id = request.args.get('content_id', type=int)
        limit = request.args.get('limit', default=5, type=int)

        if not user_id or not query_text:
            api.abort(400, "Missing required parameters: user_id and query")

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
                return {
                    'message': 'No content found for this user',
                    'results': []
                }, 200

            # Step 2: Get all embedding IDs for the user's content
            embedding_chunks = db.session.query(ContentChunk.id, ContentChunk.embedding_id)\
                .filter(ContentChunk.content_id.in_(content_ids))\
                .filter(ContentChunk.embedding_id.isnot(None))\
                .all()

            if not embedding_chunks:
                return {
                    'message': 'No embeddings found for this user\'s content',
                    'results': []
                }, 200

            # Create mapping of embedding_id to chunk_id
            embedding_to_chunk = {ec[1]: ec[0] for ec in embedding_chunks}
            embedding_ids = list(embedding_to_chunk.keys())

            # Step 3: Query embedding service for similar embeddings
            headers = {'X-API-KEY': API_KEY}
            correlation_id = request.headers.get('X-Correlation-ID')
            if correlation_id:
                headers['X-Correlation-ID'] = correlation_id
                
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
                api.abort(500, "Failed to get similar embeddings")

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

            return {
                'message': 'Search completed successfully',
                'results': results
            }, 200

        except Exception as e:
            logging.error(f"Error in search_similar_chunks: {str(e)}")
            api.abort(500, "Internal server error")

@ns.route('/content/<int:content_id>')
class ContentResource(Resource):
    @api.doc('get_content_by_id')
    @api.marshal_with(content_response)
    def get(self, content_id):
        """Get content by ID"""
        try:
            content = Content.query.get(content_id)
            if content is None:
                api.abort(404, "Content not found")
                
            return {
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
            }, 200
            
        except Exception as e:
            logging.error(f"Error in get_content_by_id: {str(e)}")
            api.abort(500, "Internal server error")

@ns.route('/chunk/<int:chunk_id>')
class ChunkResource(Resource):
    @api.doc('get_chunk_text_by_id')
    @api.marshal_with(chunk_response)
    def get(self, chunk_id):
        """Get text and embedding_id of a chunk by ID"""
        try:
            chunk = ContentChunk.query.get(chunk_id)
            if chunk is None:
                api.abort(404, "Chunk not found")
                
            return {
                'text': chunk.chunk_text,
                'embedding_id': chunk.embedding_id
            }, 200
            
        except Exception as e:
            logging.error(f"Error in get_chunk_text_by_id: {str(e)}")
            api.abort(500, "Internal server error")

@app.before_request
def log_request_info():
    # Exempt the /docs endpoint from logging and API key checks
    if request.path.startswith('/docs') or request.path.startswith('/swagger'):
        return

    logging.info(f"Headers: {request.headers}")
    logging.info(f"Body: {request.get_data()}")

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