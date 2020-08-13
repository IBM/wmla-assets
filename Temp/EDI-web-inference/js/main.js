//  Copyright 2019 IBM International Business Machines Corp. 
//  Copyright 2017 Google Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
// 
//           http://www.apache.org/licenses/LICENSE-2.0
// 
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
//  implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// The camera selection and grabbing code was heavily inspired from samdutton
// it can be found here: https://github.com/samdutton/simpl/blob/gh-pages/getusermedia/sources/js/main.js

//Globals.
var videoElement = document.querySelector('video')
var videoSelect = document.querySelector('select#videoSource')
var modeSelect = document.querySelector('select#mode')
var captureButton = document.querySelector('button#capture')
var statsButton = document.querySelector('button#GPU')
var canvas = document.querySelector('canvas')
var img = document.querySelector('img')
var authToken;
var t0, t1;
var livemode = false
var buffermode = false
var buffer = new PriorityQueue()
var config

//Main
console.log("Main")

//Get configuration.
config = new configuration_edi()

//Set menu to pick correct video stream.
videoSelect.onchange = getStream;

//Get mode and set mode for button.
//Set the button to trigger screen capturing loop.
modeSelect.onchange = getmodes

//Set GOU stats getter to the butoon
statsButton.onclick = GPUStatsLoop

//Authenticate.
getAuth()

//Display video stream. 
getStream().then(getDevices).then(gotDevices);

//end Main.

async function GPUStatsLoop() {
    while(true) {
        POSTGPUStats()
        await sleep(5000) //TODO: Should we make this a variable?
    }
}

function POSTGPUStats() {
    var xhttp = new XMLHttpRequest()
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            updateGraph(this.responseText)
        }
    }
    xhttp.open("POST", window.location.href, true)
    xhttp.send("gpustats")
}

//Start Buffer mode.
function start() {
    console.log("Start")
    captureVideoLoop()
    displayVideo()
}

//Select what mode to run the inference.
function getmodes() {
    const mode = modeSelect.value
    if(mode == 1){
        buffermode = false
        livemode = true
        captureButton.onclick = captureVideoLiveLoop
    }
    else if(mode == 2){
        buffermode = true
        livemode = false
        captureButton.onclick = start
    }
    else {
        console.log("No Mode selected.")
    }
}

//Authenticate with the Server
function getAuth() {
    var xhttp =  new XMLHttpRequest()
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            console.log("Responce "+this.responseText)
            authToken = JSON.parse(this.responseText)["user_token"]
            console.log(authToken)
         }
    };
    xhttp.open("POST", config.URL+"/dlim/v1/auth/token")
    xhttp.setRequestHeader("Content-type", "application/json")
    var json_payload = JSON.stringify({method:"password",userpasswd:config.USRPASSWD})
    //console.log(json_payload)
    xhttp.send(json_payload)
}

//Main Loop to time the screengrabs
async function captureVideoLoop() {
    while(buffermode) {
        captureVideo()
        await sleep(15);
    }
}

//Capture live video with less frames.
async function captureVideoLiveLoop() {
    while(livemode) {
        captureVideoAndDisplay()
        await sleep(150);
    }
}

//Main Display Loop
async function displayVideo() {
console.log("sleeping for 1.5")
    await sleep(1500)
console.log("done sleeping..")
    while(true){
        frame = buffer.dequeue()
        if(frame != null) {
            img.src = "data:image/jpeg;base64,"+frame["data"]
        }
        await sleep(17)
    }
}

//Capture image from camera feed and send to server for inference
function captureVideo() {
    canvas.width = videoElement.videoWidth
    canvas.height = videoElement.videoHeight
    var tstamp = performance.now()
    //console.log(tstamp)
    canvas.getContext('2d').drawImage(videoElement, 0, 0)
    //img.src = canvas.toDataURL('image/jpeg');
    var img64 = canvas.toDataURL('image/jpeg').replace(/^data:image\/(png|jpeg);base64,/, "")
    var json_payload = JSON.stringify({id:tstamp, data: img64})
    //console.log(json_payload)
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
         if (this.readyState == 4 && this.status == 200) {
             //alert(this.responseText);
             t1 = performance.now()
             //console.log("Inference request took " + (t1 - t0) + " milliseconds.");
             t0 = performance.now();
             var output = JSON.parse(this.responseText)
             //Don't display the returned image, just push onto the buffer.
             buffer.enqueue(output)
         }
         else {
             //console.log(this.responseText)
         }
    };
    //t0 = performance.now();
    xhttp.open("POST", config.URL+"/dlim/v1/inference/"+config.MODEL, true);
    xhttp.setRequestHeader("Content-type", "application/json");
    xhttp.setRequestHeader("X-User-Token", authToken)
    xhttp.send(json_payload);

}

function captureVideoAndDisplay() {
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    canvas.getContext('2d').drawImage(videoElement, 0, 0);
    //img.src = canvas.toDataURL('image/jpeg');
    var img64 = canvas.toDataURL('image/jpeg').replace(/^data:image\/(png|jpeg);base64,/, "")
    var json_payload = JSON.stringify({id:"0",data: img64})
    //console.log(json_payload)
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
         if (this.readyState == 4 && this.status == 200) {
             //alert(this.responseText);
             t1 = performance.now()
             //console.log("Inference request took " + (t1 - t0) + " milliseconds.");
             t0 = performance.now()
             var output = JSON.parse(this.responseText)
             img.src = "data:image/jpeg;base64,"+output["data"]
         }
         else {
             //console.log(this.responseText)
         }
    };
    //t0 = performance.now();
    xhttp.open("POST", config.URL+"/dlim/v1/inference/keras-yolo3", true);
    xhttp.setRequestHeader("Content-type", "application/json");
    xhttp.setRequestHeader("X-User-Token", authToken)
    xhttp.send(json_payload);

}

//Helper Functions:
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
function getDevices() {
  // AFAICT in Safari this only gets default devices until gUM is called :/
  return navigator.mediaDevices.enumerateDevices();
}

function gotDevices(deviceInfos) {
  window.deviceInfos = deviceInfos; // make available to console
  console.log('Available input and output devices:', deviceInfos);
  for (const deviceInfo of deviceInfos) {
    const option = document.createElement('option');
    option.value = deviceInfo.deviceId;
    if (deviceInfo.kind === 'videoinput') {
      option.text = deviceInfo.label || `Camera ${videoSelect.length + 1}`;
      videoSelect.appendChild(option);
    }
  }
}

function getStream() {
  if (window.stream) {
    window.stream.getTracks().forEach(track => {
      track.stop();
    });
  }
  const videoSource = videoSelect.value;
  const constraints = {
    video: {deviceId: videoSource ? {exact: videoSource} : undefined}
  };
  return navigator.mediaDevices.getUserMedia(constraints).
    then(gotStream).catch(handleError);
}

function gotStream(stream) {
  window.stream = stream; // make stream available to console
  videoSelect.selectedIndex = [...videoSelect.options].
    findIndex(option => option.text === stream.getVideoTracks()[0].label);
  videoElement.srcObject = stream;
}

function handleError(error) {
  console.error('Error: ', error);
}

