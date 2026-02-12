import http.server
import socketserver
import os
from datetime import datetime

PORT = 8000
LOG_FILE = "debug_log.txt"

print(f"Iniciando servidor na porta {PORT}...")
print(f"Logs salvos em: {os.path.abspath(LOG_FILE)}")

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/log':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                message = post_data.decode('utf-8')
                
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                log_entry = f"{timestamp} {message}\n"
                
                with open(LOG_FILE, "a") as f:
                    f.write(log_entry)
                
                print(f"LOG: {message}")
                
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b"OK")
            except Exception as e:
                print(f"Erro ao processar log: {e}")
                self.send_response(500)
                self.end_headers()
        else:
            self.send_error(404)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

socketserver.TCPServer.allow_reuse_address = True

with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
    print("Servidor pronto. Aguardando logs...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
