import { Handler } from '@netlify/functions';
import { MongoClient } from 'mongodb';

// Hardcoded as requested
const MONGODB_URI = 'mongodb+srv://datacollector:datacollector67@cluster0.vc6vmf8.mongodb.net/?appName=Cluster0';
const DB_NAME = 'pose_data';
const COLLECTION_NAME = 'captures';

let cachedClient: MongoClient | null = null;

function decodeNaNPlaceholders(value: unknown): unknown {
  if (value === 'NaN') return Number.NaN;
  if (Array.isArray(value)) return value.map(decodeNaNPlaceholders);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [
        key,
        decodeNaNPlaceholders(item),
      ])
    );
  }
  return value;
}

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
  // GET request acts as a connection test
  if (event.httpMethod === 'GET') {
    try {
      const client = await connectToDatabase();
      await client.db(DB_NAME).command({ ping: 1 });
      return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ success: true, message: 'MongoDB connection successful!' }),
      };
    } catch (error) {
      console.error('MongoDB test error:', error);
      return {
        statusCode: 500,
        headers: { 'Content-Type': 'application/json' },
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
      body: JSON.stringify({ message: 'Method Not Allowed' }),
    };
  }

  try {
    const payload = decodeNaNPlaceholders(JSON.parse(event.body || '{}')) as any;
    const { captureId, ...data } = payload;

    if (!captureId || !data.meta) {
      return {
        statusCode: 400,
        body: JSON.stringify({ message: 'Missing captureId or metadata' }),
      };
    }

    const client = await connectToDatabase();
    const db = client.db(DB_NAME);
    const collection = db.collection(COLLECTION_NAME);

    // Prepare the document to insert
    const document = {
      capture_id: captureId,
      ...data,
      created_at: new Date().toISOString(),
    };

    // Upsert the document
    await collection.updateOne(
      { capture_id: captureId },
      { $set: document },
      { upsert: true }
    );

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ success: true, message: 'Successfully pushed to MongoDB' }),
    };
  } catch (error) {
    console.error('MongoDB push error:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ 
        message: 'Internal Server Error', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      }),
    };
  }
};
