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

//PriorityQueue class to act as main buffer.
class PriorityQueue {

    //An array is used to implement priority
    constructor() 
    {
        this.items = [];
    }
    //Add element to list.
    enqueue(element)
    { 
        var added = false
        //Id is greater than last element, add last.
        if(this.items.length > 0 && this.items[this.items.length - 1]["key"] < element["key"]) {
            this.items.push(element)
        }
        else {
            for (var i = 0 ; i < this.items.length; i++) { 
                if (element["key"] < this.items[i]["key"] ) { 
                    // Once the correct location is found it is 
                    // enqueued 
                    this.items.splice(i, 0, element);
                    added = true
                    console.log("OOO!!!")
                    break;
                } 
            } 

            // This means the queue is empty, or something weird is going on...
            if (!added) { 
                this.items.push(element)
                console.log("Added last...")
            } 
        }
    }
    // Get next on list.
    dequeue()
    {
        if(this.items.length == 0){
            //No more.
            return null
        }
        return this.items.shift()
    }
}
