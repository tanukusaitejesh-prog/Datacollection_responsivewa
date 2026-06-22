import http from 'http';
import { MongoClient } from 'mongodb';

const MONGODB_URI = 'mongodb+srv://datacollector:datacollector67@cluster0.vc6vmf8.mongodb.net/?appName=Cluster0';
const DB_NAME = 'pose_data';
const COLLECTION_NAME = 'captures';

let cachedClient = null;
async function connectToDatabase() {
  if (cachedClient) return cachedClient;
  const client = new MongoClient(MONGODB_URI);
  await client.connect();
  cachedClient = client;
  return client;
}

const server = http.createServer(async (req, res) => {
  // Add CORS headers just in case
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // Parse path
  const parsedUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const pathname = parsedUrl.pathname;

  if (pathname === '/.netlify/functions/pushToMongo') {
    if (req.method === 'GET') {
      try {
        const client = await connectToDatabase();
        await client.db(DB_NAME).command({ ping: 1 });
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, message: 'MongoDB connection successful!' }));
      } catch (error) {
        console.error('MongoDB test error:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: error.message }));
      }
    } else if (req.method === 'POST') {
      let body = '';
      req.on('data', chunk => {
        body += chunk.toString();
      });
      req.on('end', async () => {
        try {
          const payload = JSON.parse(body || '{}');
          const { captureId, npy_file, npy_file_pose_hands, ...data } = payload;

          if (!captureId || !data.meta) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ message: 'Missing captureId or metadata' }));
            return;
          }

          const client = await connectToDatabase();
          const db = client.db(DB_NAME);
          const collection = db.collection(COLLECTION_NAME);

          const document = {
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

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true, message: 'Successfully pushed to MongoDB' }));
        } catch (error) {
          console.error('MongoDB push error:', error);
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ message: 'Internal Server Error', error: error.message }));
        }
      });
    } else {
      res.writeHead(405, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ message: 'Method Not Allowed' }));
    }
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ message: 'Not Found' }));
  }
});

const PORT = 8888;
server.listen(PORT, () => {
  console.log(`Local Netlify functions emulator running at http://localhost:${PORT}`);
});
