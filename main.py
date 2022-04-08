# Copyright 2022 Seggan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.signal import find_peaks
import librosa
import math

"""
Opcodes:
0  - NOP
22 - NOP
"""

class Stack:
  idx = 0
  stack = []
  def __init__(self, opcodes) -> None:
      self.opcodes = opcodes
  def tick(self):
    opcode = self.opcodes[self.idx]

file = "a.wav"
# length (in seconds) of each bit we will process
opsec_size = 1
data, sr = librosa.load(file)

n_fft=2048
spec = np.abs(librosa.stft(data, n_fft=n_fft))
spec /= spec.max()
spec *= 22
spec = np.floor(spec)
spec = spec.astype(int)

with open("out.txt", "w") as f:
  for i in range(len(spec)):
    for j in range(len(spec[i])):
      f.write(str(spec[i][j]) + " ")
    f.write("\n")

sums = [sum(x) for x in spec]
peaks = find_peaks(sums)[0]
stack_info = [spec[f].tolist() for f in peaks]

time = len(data) / sr

grouped = []
for stack in stack_info:
    time_per_bucket = time / len(stack)
    buckets_per_opsec = opsec_size / time_per_bucket
    is_whole_number = math.isclose(buckets_per_opsec, round(buckets_per_opsec))

    res = []
    sum = []
    i = 0
    while len(stack) > 0:
        item = stack.pop(0)
        sum.append(item)
        if i != 0 and i % math.floor(buckets_per_opsec) == 0:
            if not is_whole_number and len(stack) > 0:
                sum.append(stack[0])
            res.append(np.floor(np.mean(sum)).astype(int))
            sum = []
        i += 1
    
    grouped.append(res)

print(grouped)

