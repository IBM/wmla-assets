# HTML5 Web Inference demo. 

### Prereqs.

* A running instance of WML-A with EDI.
* An HTML5 capable device/browser.
* The YOLOv3 infernce model running on EDI.

### Configuration

To configure the client please edit the values under `config.js`.
You will need:
* The full URL and port the EDI REST API endpoint. Such as `https://mydomain.com:9000`
* The username and password to authenticate to EDI with. You can generate it with this command: `echo '<username>:<password>' | base64`
* The name of the deployed model. Example: `keras-yolo3`

### Getting started.

On your web browser navigate to the location of the demo folder. 
Such as `file:///home/jpvillam/demo/index.html`

After allowing permissions to the browser to access the camera simply choose a mode and then capture video!

### Modes.

* Live!

Live mode tries to capture your camera output in real time, then it sends requests to the server for image recognition.
Due to network inconsistencies we try to avoid out of order frames, so we keep the FPS relatively low at a little less than 7 FPS.

* Buffer

In order to achieve a more smooth looking detection we create a buffer to correct network inconsistencies. This causes a delay
in the recording, but we can get much better FPS at around 40 FPS.

### Serving to other devices.
Note that the `Get GPU stats` button will not work withour this. 

Install python simple http server with: `pip3 install httpserver`
In this demo I include a very simple webserver, to configure replace the `server_address` in `server.py` to the 
correct IP address of the machine this is running in.

To access camera devices the website must be served over https, so we need to generate an ssl certificate.
To generate a quick one use `openssl req -new -x509 -keyout localhost.pem -out localhost.pem -days 365 -nodes`
in the same location as the `server.py` file. 

Simply run the server by the command `python3 server.py`

You are ready to access the website from another device in the local network, simple go to `https://<server-ip>:4004`
