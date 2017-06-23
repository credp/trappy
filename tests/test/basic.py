#    Copyright 2016-2017 ARM Limited
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
#
import trappy
import re

my_events = [
    "sched_switch",
    "sched_wakeup",
    "sched_contrib_scale_f",
    "sched_load_avg_cpu",
    "sched_load_avg_task",
    "sched_tune_tasks_update",
    "sched_boost_cpu",
    "sched_boost_task",
    "sched_energy_diff",
    "sched_overutilized",
    "cpu_frequency",
    "cpu_capacity",
    "cpu_idle"
]
trace=trappy.FTrace(events=my_events)

print "\n\nPARSING DONE\n\nChecking..."

# build a dict of trace lines with their timestamps as keys
SPECIAL_FIELDS_RE = re.compile(
                        r"^\s*(?P<comm>.*)-(?P<pid>\d+)(?:\s+\(.*\))"\
                        r"?\s+\[(?P<cpu>\d+)\](?:\s+....)?\s+"\
                        r"(?P<timestamp>[0-9]+(?P<us>\.[0-9]+)?): (\w+:\s+)+(?P<data>.+)" )

timestamps = {}
missing_traces = {}
with open("trace.txt") as fin:
    lines = fin.readlines()
for line in lines:
    fields_match = SPECIAL_FIELDS_RE.match(line)
    if not fields_match:
        continue
    timestamp = float(fields_match.group('timestamp'))
    if not fields_match.group('us'):
        timestamp /= 1e9
    timestamps[timestamp] = [ line, None ]

got_lines = 0
unique_words = []
for t in trace.trace_classes:
    unique_words.append(t.unique_word)
    df = t.data_frame
    timestamps_in_df = df.index.values
    got_lines += len(timestamps_in_df)
    for t in timestamps_in_df:
        timestamps[t][1] = 'seen'

print unique_words

print "\n"
lost_lines = 0
for key in timestamps:
    line = timestamps[key][0]
    seen = timestamps[key][1]
    if not seen:
        l = line.split()
        print "{} not found ({})".format(key, l)
        idx=None
        for i in range(len(l)):
            if l[i] not in unique_words:
                continue
            print "found {} in unique words".format(l[i])
            idx = i
            break
                
        if idx is not None:
            if l[idx] not in missing_traces:
                missing_traces[l[idx]] = 0
            missing_traces[l[idx]] += 1
        lost_lines += 1

print "\n\nGot {} lines.".format(got_lines)
print "Lost {} lines.".format(lost_lines)
for key in missing_traces:
    print "{} - {}".format(key, missing_traces[key])