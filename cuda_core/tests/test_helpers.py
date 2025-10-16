from cuda.core.experimental import Device
from helpers.latch import LatchKernel
from memory_ipc.utility import TimestampedLogger, make_scratch_buffer, compare_buffers
import time

ENABLE_LOGGING = False  # Set True for test debugging and development
NBYTES = 64

def test_latchkernel():
    log = TimestampedLogger()
    log("begin")
    device = Device()
    device.set_current()
    stream = device.create_stream()
    target = make_scratch_buffer(device, 0, NBYTES)
    zeros = make_scratch_buffer(device, 0, NBYTES)
    ones = make_scratch_buffer(device, 1, NBYTES)
    latch = LatchKernel(device)
    log("launching latch kernel")
    latch.launch(stream)
    log("launching copy (0->1) kernel")
    target.copy_from(ones, stream=stream)
    log("going to sleep")
    time.sleep(1)
    log("checking target == 0")
    assert compare_buffers(target, zeros) == 0
    log("releasing latch and syncing")
    latch.release()
    stream.sync()
    log("checking target == 1")
    assert compare_buffers(target, ones) == 0
    log("done")

