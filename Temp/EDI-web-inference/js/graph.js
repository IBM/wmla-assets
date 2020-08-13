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

var labels = []
//TODO: support more than 4 GPUs/Across machine :/ 
var dataset = [{ 
        data: [],
        label: "gpu0",
        backgroundColor: "rgba(62, 149, 205, 0.5)",
        order: 1
      }, { 
        data: [],
        label: "gpu1",
        backgroundColor: "rgba(142, 94, 162, 0.5)",
        order: 2
      }, {
        data: [],
        label: "gpu2",
        backgroundColor: "rgba(60, 186, 159, 0.5)",
        order: 3
      }, {
        data: [],
        label: "gpu3",
        backgroundColor: "rgba(232, 195, 185, 0.5)",
        order: 4
      }, {
        label: "Average",
        backgroundColor: '#000000',
        borderColor: '#000000',
        data:[],
        order: 0,
        fill: false,
        type: 'line'
      }]

config = new configuration_edi()
var ctx = document.getElementById('GPUGraph').getContext('2d')
var gpusGraph = null
var max_plot = config.MAXPLOT
var cur_plot = 0

function generateLabels(){
    var label_limit = 0
    var i

    labels.push("t:"+cur_plot)

    //What is the starting number.
    if(cur_plot >= max_plot) {
        //Past limit shift one off.
        labels.shift()
    }
    else {
        //Haven't reached limit.
    }

    return labels
}

function generateData(raw_data) {
    var avg = 0
    //Remove 3 character off the front and back, remove all spaces, and split on \n
    array_data = raw_data.substring(3, raw_data.length-3).replace(/\s/g, '').split("\\n")
    for(var i=0; i < array_data.length; i++) { 
        dataset[i].data.push(array_data[i])
        avg += parseInt(array_data[i])
        if(cur_plot >= max_plot){
            dataset[i].data.shift()
        }
    }

    // Add the average to the line dataset.
    dataset[4].data.push(avg/4)
    if(cur_plot >= max_plot){
        dataset[4].data.shift()
    }
    return dataset
}

function updateGraph(raw_data) {

    //If no maximum amount of data, plot forever.
    if(config.MAXPLOT == 0) {
        max_plot++
    }
    //Generate everything on the fly, because we cool like that
    generateData(raw_data)
    generateLabels()

    //Increase current "time"
    cur_plot++
    
    if(gpusGraph == null) {
        gpusGraph = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: dataset
            },
	    options: {maintainAspectRatio: false}
        })
    }
    else {
        gpusGraph.data.labels = labels
        gpusGraph.data.datasets = dataset
        gpusGraph.update()
    }
    
}
