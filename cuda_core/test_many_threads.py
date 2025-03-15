"""Experiment:
$ python test_many_threads.py | sort | uniq -c
   1000 COMPUTE _tls.devices END
   1000 COMPUTE _tls.devices START
     10 REPEAT END
     10 REPEAT START
"""

import threading

from cuda.core.experimental import Device


def run_with_threads(target_func, num_threads=100, repeats=10):
    for _ in range(repeats):
        print("REPEAT START", flush=True)
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=target_func)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        print("REPEAT END", flush=True)


run_with_threads(Device)
