import { Handler } from '@netlify/functions';
import { MongoClient } from 'mongodb';

// Environment variables for security
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb+srv://datacollector:datacollector67@cluster0.vc6vmf8.mongodb.net/?appName=Cluster0';
const DB_NAME = process.env.MONGODB_DB_NAME || 'pose_data';
const COLLECTION_NAME = process.env.MONGODB_COLLECTION || 'captures';

let cachedClient: MongoClient | null = null;

async function connectToDatabase() {
  if (cachedClient) {
    return cachedClient;
  }
  const client = new MongoClient(MONGODB_URI);
  await client.connect();
  cachedClient = client;
  return client;
}

export const handler: Handler = async (event) => {
  const headers = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  };

  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 204, headers, body: '' };
  }

  // GET request acts as a connection test
  if (event.httpMethod === 'GET') {
    try {
      const client = await connectToDatabase();
      await client.db(DB_NAME).command({ ping: 1 });
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ success: true, message: 'MongoDB connection successful!' }),
      };
    } catch (error) {
      console.error('MongoDB test error:', error);
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ 
          success: false, 
          error: error instanceof Error ? error.message : 'Unknown connection error' 
        }),
      };
    }
  }

  // Only allow POST and GET
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ message: 'Method Not Allowed' }),
    };
  }

  try {
    const payload = JSON.parse(event.body || '{}');
    const { captureId, chunk, npy_file, npy_file_pose_hands, ...data } = payload;

    if (!captureId) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ message: 'Missing captureId' }),
      };
    }

    const client = await connectToDatabase();
    const db = client.db(DB_NAME);
    const collection = db.collection(COLLECTION_NAME);

    // --- Chunked upload support ---
    // chunk === 'meta'      → upsert metadata + keypoints (no binary)
    // chunk === 'npy_pose'  → append npy_file binary
    // chunk === 'npy_hands' → append npy_file_pose_hands binary
    // chunk === undefined   → legacy single-shot upload (backward compat)

    if (chunk === 'meta') {
      // First chunk: metadata + keypoints
      const document: any = {
        capture_id: captureId,
        ...data,
        created_at: new Date().toISOString(),
      };
      await collection.updateOne(
        { capture_id: captureId },
        { $set: document },
        { upsert: true }
      );
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ success: true, message: 'Metadata saved' }),
      };
    }

    if (chunk === 'npy_pose' && npy_file) {
      await collection.updateOne(
        { capture_id: captureId },
        { $set: { npy_file: Buffer.from(npy_file, 'base64') } }
      );
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ success: true, message: 'NPY pose file saved' }),
      };
    }

    if (chunk === 'npy_hands' && npy_file_pose_hands) {
      await collection.updateOne(
        { capture_id: captureId },
        { $set: { npy_file_pose_hands: Buffer.from(npy_file_pose_hands, 'base64') } }
      );
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ success: true, message: 'NPY hands file saved' }),
      };
    }

    // --- Legacy single-shot upload (backward compat) ---
    if (!data.meta) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ message: 'Missing captureId or metadata' }),
      };
    }

    const document: any = {
      capture_id: captureId,
      ...data,
      created_at: new Date().toISOString(),
    };

    if (npy_file) {
      document.npy_file = Buffer.from(npy_file, 'base64');
    }
    if (npy_file_pose_hands) {
      document.npy_file_pose_hands = Buffer.from(npy_file_pose_hands, 'base64');
    }

    await collection.updateOne(
      { capture_id: captureId },
      { $set: document },
      { upsert: true }
    );

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ success: true, message: 'Successfully pushed to MongoDB' }),
    };
  } catch (error) {
    console.error('MongoDB push error:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ 
        message: 'Internal Server Error', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      }),
    };
  }
};
