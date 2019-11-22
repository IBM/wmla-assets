#  copyright 2019 ibm international business machines corp.
#
#  licensed under the apache license, version 2.0 (the "license");
#  you may not use this file except in compliance with the license.
#  you may obtain a copy of the license at
# 
#           http://www.apache.org/licenses/license-2.0
# 
#  unless required by applicable law or agreed to in writing, software
#  distributed under the license is distributed on an "as is" basis,
#  without warranties or conditions of any kind, either express or
#  implied.
#  see the license for the specific language governing permissions and
#  limitations under the license.
import http.server, ssl
import subprocess
from io import BytesIO


class GPUHandler(http.server.SimpleHTTPRequestHandler):
    """Extend SimpleHTTPRequestHandler to return GPU stats on POST."""

    def do_POST(self):
        """POST GPU Stats""" 
        content_length = int(self.headers['content-length'])
        content_data = self.rfile.read(content_length)
        #print("Content:")
        #print(content_data)
        self.send_response(200)
        self.end_headers()
        p = subprocess.Popen("nvidia-smi | grep % | cut -c 62-63", stdout=subprocess.PIPE, shell=True)
        stats, err = p.communicate()
        stats = str(stats).replace('\n', ' ')
        response = BytesIO()
        response.write(stats.encode())
        self.wfile.write(response.getvalue())

server_address = ('9.3.89.146', 4004)
httpd = http.server.HTTPServer(server_address, GPUHandler)
httpd.socket = ssl.wrap_socket(httpd.socket,
                               server_side=True,
                               certfile='localhost.pem',
                               ssl_version=ssl.PROTOCOL_TLSv1)
print("Serving")
httpd.serve_forever()
