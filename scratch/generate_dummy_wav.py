import wave
import math
import struct

sample_rate = 16000
duration = 1.0
num_samples = int(sample_rate * duration)

with wave.open('scratch/dummy.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    for i in range(num_samples):
        val = int(32767 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
        data = struct.pack('<h', val)
        f.writeframesraw(data)
