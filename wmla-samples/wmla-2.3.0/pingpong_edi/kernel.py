#!/usr/bin/env python

import redhareapiversion
from redhareapi import Kernel
import json

class TestKernel(Kernel):

    def on_task_invoke(self, task_context):
        try:
            Kernel.log_debug("on_task_invoke")
            # using loop to handle batch-size
            while task_context != None:
                Kernel.log_debug("task id: %s " % task_context.get_id())
                input_data = json.loads(task_context.get_input_data())
                if 'seq' in input_data:
                    Kernel.log_debug("sequence number: %s " % input_data['seq'])
                task_context.set_output_data(json.dumps(input_data))
                task_context = task_context.next()
        except Exception as e:
            Kernel.log_error(str(e))

if __name__ == '__main__':
    ppkernel = TestKernel()
    ppkernel.run()
