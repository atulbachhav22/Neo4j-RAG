-- Neo4j database setup queries
-- Run these in Neo4j Browser or using cypher-shell

-- Create constraints and indexes for better performance
CREATE CONSTRAINT document_filename_unique IF NOT EXISTS 
FOR (d:Document) REQUIRE d.filename IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS 
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

-- Create indexes for better search performance
CREATE INDEX chunk_content_fulltext IF NOT EXISTS 
FOR (c:Chunk) ON (c.content);

CREATE INDEX document_filename_index IF NOT EXISTS 
FOR (d:Document) ON (d.filename);

-- Create vector index for embeddings (requires Neo4j 5.11+)
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
};

-- Sample query patterns for the knowledge graph

-- Find similar chunks
MATCH (c1:Chunk)
CALL db.index.vector.queryNodes('chunk_embeddings', 5, c1.embedding)
YIELD node AS c2, score
WHERE c1 <> c2
RETURN c1.content, c2.content, score;

-- Find documents with most chunks
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
RETURN d.filename, count(c) as chunk_count
ORDER BY chunk_count DESC;

-- Create semantic relationships based on similarity
MATCH (c1:Chunk), (c2:Chunk)
WHERE c1.id < c2.id
WITH c1, c2, gds.similarity.cosine(c1.embedding, c2.embedding) AS similarity
WHERE similarity > 0.8
CREATE (c1)-[:SIMILAR_TO {similarity: similarity}]->(c2);

-- Find clusters of related content
MATCH path = (c1:Chunk)-[:SIMILAR_TO*1..3]-(c2:Chunk)
RETURN path LIMIT 10;