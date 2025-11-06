import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { ServerResponse } from 'http';

// https://vitejs.dev/config/
export default defineConfig({
  // Set the base path for deployment to the specific GitHub repository.
  base: '/INF3/',
  plugins: [react()],
  server: {
    watch: {
      // Ignore changes to tsconfig.json to prevent unwanted server restarts
      // if an IDE or another tool modifies it.
      ignored: ['**/tsconfig.json'],
    },
    proxy: {
      // 本地后端服务代理（知识库和AI生成请求）
      '/api': {
        target: 'http://127.0.0.1:5000', // 本地开发时指向本地Flask后端
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        // 添加日志以便调试代理问题
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log(`[vite-proxy] Sending request: ${req.method} ${req.url} -> ${options.target}${proxyReq.path}`);
          });
          proxy.on('proxyRes', (proxyRes, req, res) => {
            console.log(`[vite-proxy] Received response: ${proxyRes.statusCode} ${req.url}`);
          });
          proxy.on('error', (err, req, res) => {
            console.error('[vite-proxy] Error:', err);
            // FIX: Use the imported ServerResponse for a proper type guard to resolve TypeScript errors on `res` which could be a Socket.
            if (res instanceof ServerResponse && !res.headersSent) {
              res.writeHead(502, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ message: 'Proxy Error', error: err.message }));
            }
          });
        }
      },
      // OpenAI API 代理
      '/proxy/my-openai': {
        target: 'https://api.chatanywhere.tech',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/proxy\/my-openai/, ''),
      },
      // DeepSeek API 代理
      '/proxy/deepseek': {
        target: 'https://api.deepseek.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/proxy\/deepseek/, ''),
      },
      // 阿里云（豆包）API 代理
      '/proxy/ali': {
        target: 'https://www.dmxapi.cn',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/proxy\/ali/, ''),
      },
    }
  },
})
