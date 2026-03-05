/**
 * Express server that serves static files and provides lightx2v result_url proxy API.
 * Railway runs npm start without a shell; PORT from env.
 */
import express from 'express';
import { createServer } from 'http';
import handler from 'serve-handler';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const port = process.env.PORT || '3000';


// Serve static files from dist
app.use((req, res) => {
  return handler(req, res, {
    public: path.join(__dirname, 'dist'),
    cleanUrls: true,
    rewrites: [{ source: '**', destination: '/index.html' }]
  });
});

const server = createServer(app);
server.listen(parseInt(port, 10) || 3000, '0.0.0.0', () => {
  console.log(`Server listening on port ${port}`);
});
server.on('error', (err) => {
  console.error('Server error:', err);
  process.exit(1);
});
