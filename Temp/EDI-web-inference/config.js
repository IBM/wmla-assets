//  Copyright 2019 IBM International Business Machines Corp. 
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

//Configuration for demo.
class configuration_edi {

	//Change values to point to correct environment. 
	constructor()
	{
	  this.URL = "https://wmla-11.aus.stglabs.ibm.com:9000" //Full URL and Port for EDI Rest API.
	  this.USRPASSWD = "QWRtaW46QWRtaW4=" //64bit encoded username and password to authenticate.
	  this.MODEL = "keras-yolo3" //Name of the inference model to make requests to.
	  this.MAXPLOT = 10 //Max number of datapoints for the graph, zero is infinite.
	}
}
