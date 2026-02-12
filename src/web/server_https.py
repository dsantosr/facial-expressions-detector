#!/usr/bin/env python3

import http.server
import ssl
import os
import subprocess
import sys
import socketserver

PORT = 8443

def generate_certificate():
    cert_file = 'localhost.pem'
    key_file = 'localhost-key.pem'
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print(f"Usando certificado: {cert_file}")
        return cert_file, key_file
    
    print("Gerando certificado...")
    subprocess.run([
        'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
        '-keyout', key_file,
        '-out', cert_file,
        '-days', '365',
        '-nodes',
        '-subj', '/CN=localhost'
    ], check=True)
    
    print(f"Certificado gerado: {cert_file}")
    return cert_file, key_file


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        if self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return
        if self.path.startswith('/.well-known'):
            self.send_error(404, "Not Found")
            return
        super().do_GET()

    def log_error(self, format, *args):
        # Prevent logging errors for common missing/ignored files to reduce noise
        if any(x in self.path for x in ['favicon.ico', '.well-known']):
             return
        super().log_error(format, *args)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # Silently handle common SSL/Connection errors that clutter the console
        exc_type, exc_value, _ = sys.exc_info()
        if exc_type in [ssl.SSLEOFError, BrokenPipeError, ConnectionResetError]:
            return
        super().handle_error(request, client_address)


def run_server():
    web_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_dir)
    print(f"Servindo de: {web_dir}")
    
    cert_file, key_file = generate_certificate()
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    
    httpd = ThreadingHTTPServer(('0.0.0.0', PORT), NoCacheHTTPRequestHandler)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    print("="*60)
    print("SERVIDOR HTTPS RODANDO (MULTI-THREADED)")
    print("="*60)
    print(f"   https://<IP>:{PORT}")
    print(f"   1. Aceite o aviso do certificado")
    print(f"   2. Permita acesso à câmera")
    print("\nPressione Ctrl+C para parar")
    print("="*60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServidor encerrado.")
        httpd.shutdown()

if __name__ == "__main__":
    run_server()
