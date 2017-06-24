#    Copyright 2015-2017 ARM Limited
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


# pylint can't see any of the dynamically allocated classes of FTrace
# pylint: disable=no-member

import itertools
import os
import sys
import re
import pandas as pd
import multiprocessing
import time

from trappy.bare_trace import BareTrace
from trappy.utils import listify
from types import ListType

class FTraceParseError(Exception):
    pass

def _plot_freq_hists(allfreqs, what, axis, title):
    """Helper function for plot_freq_hists

    allfreqs is the output of a Cpu*Power().get_all_freqs() (for
    example, CpuInPower.get_all_freqs()).  what is a string: "in" or
    "out"

    """
    import trappy.plot_utils

    for ax, actor in zip(axis, allfreqs):
        this_title = "freq {} {}".format(what, actor)
        this_title = trappy.plot_utils.normalize_title(this_title, title)
        xlim = (0, allfreqs[actor].max())

        trappy.plot_utils.plot_hist(allfreqs[actor], ax, this_title, "KHz", 20,
                             "Frequency", xlim, "default")

SPECIAL_FIELDS_RE = re.compile(
                        r"^\s*(?P<comm>.*)-(?P<pid>\d+)(?:\s+\(.*\))"\
                        r"?\s+\[(?P<cpu>\d+)\](?:\s+....)?\s+"\
                        r"(?P<timestamp>[0-9]+(?P<us>\.[0-9]+)?): (\w+:\s+)+(?P<data>.+)"
)

class GenericFTrace(BareTrace):
    """Generic class to parse output of FTrace.  This class is meant to be
subclassed by FTrace (for parsing FTrace coming from trace-cmd) and SysTrace."""

    thermal_classes = {}

    sched_classes = {}

    dynamic_classes = {}

    def __init__(self, name="", input_lines=None, normalize_time=True, scope="all",
                 events=[], window=(0, None), abs_window=(0, None),
                 num_processes=1, block_len=10000):
        super(GenericFTrace, self).__init__(name)

        self.multiprocess_count = num_processes
        self.block_len = block_len
        self.input_lines = input_lines

        self.class_definitions.update(self.dynamic_classes.items())
        self.__add_events(listify(events))

        if scope == "thermal":
            self.class_definitions.update(self.thermal_classes.items())
        elif scope == "sched":
            self.class_definitions.update(self.sched_classes.items())
        elif scope != "custom":
            self.class_definitions.update(self.thermal_classes.items() +
                                          self.sched_classes.items())

        for attr, class_def in self.class_definitions.iteritems():
            trace_class = class_def()
            setattr(self, attr, trace_class)
            self.trace_classes.append(trace_class)

        # save parameters to complete init later
        self.normalize_time = normalize_time
        self.window = window
        self.abs_window = abs_window
        self.requested_events = events

    @classmethod
    def register_parser(cls, cobject, scope):
        """Register the class as an Event. This function
        can be used to register a class which is associated
        with an FTrace unique word.

        .. seealso::

            :mod:`trappy.dynamic.register_dynamic_ftrace` :mod:`trappy.dynamic.register_ftrace_parser`

        """

        if not hasattr(cobject, "name"):
            cobject.name = cobject.unique_word.split(":")[0]

        # Add the class to the classes dictionary
        if scope == "all":
            cls.dynamic_classes[cobject.name] = cobject
        else:
            getattr(cls, scope + "_classes")[cobject.name] = cobject

    @classmethod
    def unregister_parser(cls, cobject):
        """Unregister a parser

        This is the opposite of FTrace.register_parser(), it removes a class
        from the list of classes that will be parsed on the trace

        """

        # TODO: scopes should not be hardcoded (nor here nor in the FTrace object)
        all_scopes = [cls.thermal_classes, cls.sched_classes,
                      cls.dynamic_classes]
        known_events = ((n, c, sc) for sc in all_scopes for n, c in sc.items())

        for name, obj, scope_classes in known_events:
            if cobject == obj:
                del scope_classes[name]

    def _do_parse(self):
        should_finalize = True
        if not self.input_data:
            self.input_data = self.trace_path
        else:
            should_finalize = False
        self.__parse_trace_data(self.input_data, self.window, self.abs_window)

        if not should_finalize:
            return

        self.finalize_objects()

        if self.normalize_time:
            self._normalize_time()

    def __add_events(self, events):
        """Add events to the class_definitions

        If the events are known to trappy just add that class to the
        class definitions list.  Otherwise, register a class to parse
        that event

        """

        from trappy.dynamic import DynamicTypeFactory, default_init
        from trappy.base import Base

        # TODO: scopes should not be hardcoded (nor here nor in the FTrace object)
        all_scopes = [self.thermal_classes, self.sched_classes,
                      self.dynamic_classes]
        known_events = {k: v for sc in all_scopes for k, v in sc.iteritems()}

        for event_name in events:
            for cls in known_events.itervalues():
                if (event_name == cls.unique_word) or \
                   (event_name + ":" == cls.unique_word):
                    self.class_definitions[event_name] = cls
                    break
            else:
                kwords = {
                    "__init__": default_init,
                    "unique_word": event_name + ":",
                    "name": event_name,
                }
                trace_class = DynamicTypeFactory(event_name, (Base,), kwords)
                self.class_definitions[event_name] = trace_class

    def __populate_data(self, fin, cls_for_unique_word):
        """Append to trace data from a txt trace"""

        def contains_unique_word(line, unique_words=cls_for_unique_word.keys()):
            for unique_word in unique_words:
                if unique_word in line:
                    return True
            return False

        actual_trace = itertools.dropwhile(self.trace_hasnt_started(), fin)
        actual_trace = itertools.takewhile(self.trace_hasnt_finished(),
                                           actual_trace)


        for line in actual_trace:
            #print '.',
            if not contains_unique_word(line):
                self.lines += 1
                continue
            for unique_word, cls in cls_for_unique_word.iteritems():
                if unique_word in line:
                    trace_class = cls
                    if not cls.fallback:
                        break
            else:
                if not trace_class:
                    raise FTraceParseError("No unique word in '{}'".format(line))

            line = line[:-1]

            fields_match = SPECIAL_FIELDS_RE.match(line)
            if not fields_match:
                raise FTraceParseError("Couldn't match fields in '{}'".format(line))
            comm = fields_match.group('comm')
            pid = int(fields_match.group('pid'))
            cpu = int(fields_match.group('cpu'))

            # The timestamp, depending on the trace_clock configuration, can be
            # reported either in [s].[us] or [ns] format. Let's ensure that we
            # always generate DF which have the index expressed in:
            #    [s].[decimals]
            timestamp = float(fields_match.group('timestamp'))
            if not fields_match.group('us'):
                timestamp /= 1e9
            data_str = fields_match.group('data')

            if not self.basetime:
                self.basetime = timestamp

            if (timestamp < self.window[0] + self.basetime) or \
               (timestamp < self.abs_window[0]):
                self.lines += 1
                #print "-",
                continue

            if (self.window[1] and timestamp > self.window[1] + self.basetime) or \
               (self.abs_window[1] and timestamp > self.abs_window[1]):
                return

            # Remove empty arrays from the trace
            data_str = re.sub(r"[A-Za-z0-9_]+=\{\} ", r"", data_str)

            trace_class.append_data(timestamp, comm, pid, cpu, self.lines, data_str)
            self.lines += 1
            #print "+",

    def trace_hasnt_started(self):
        """Return a function that accepts a line and returns true if this line
            is not part of the trace.

        Subclasses of GenericFTrace may override this to skip the
        beginning of a file that is not part of the trace.  The first
        time the returned function returns False it will be considered
        the beginning of the trace and this function will never be
        called again (because once it returns False, the trace has
        started).

        """
        return lambda line: not SPECIAL_FIELDS_RE.match(line)

    def trace_hasnt_finished(self):
        """Return a function that accepts a line and returns true if this line
            is part of the trace.

        This function is called with each line of the file *after*
        trace_hasnt_started() returns True so the first line it sees
        is part of the trace.  The returned function should return
        True as long as the line it receives is part of the trace.  As
        soon as this function returns False, the rest of the file will
        be dropped.  Subclasses of GenericFTrace may override this to
        stop processing after the end of the trace is found to skip
        parsing the end of the file if it contains anything other than
        trace.

        """
        return lambda x: True

    def __memoize_unique_words(self):
        # Memoize the unique words to speed up parsing the trace file
        cls_for_unique_word = {}
        for trace_name in self.class_definitions.iterkeys():
            trace_class = getattr(self, trace_name)

            unique_word = trace_class.unique_word
            cls_for_unique_word[unique_word] = trace_class
        return cls_for_unique_word

    # This function is called in the dispatcher process when multiprocessing.
    # It is responsible for getting a blob of data returned from a worker
    # and putting it into the dispatcher object as if it had been parsed locally.
    # The workers can dispatch data back as either dataframes or tuples of the
    # pre-dataframe data arrays since it is not clear yet which is better.
    def __receive_data_blob(self, data_blob):
        # The data blob is always a tuple, but the 3rd value
        # can be either another tuple (arrays) or a dataframe
        trace_class_name, line_base, embedded_blob = data_blob
        if isinstance(embedded_blob, tuple):
            # Data was sent back as a collection of arrays in a tuple
            # unpack them and insert them into the object as if they had
            # come directly from a trace file.
            for t in self.trace_classes:
                if t.name == trace_class_name:
                    ( time_array, line_array, comm_array,
                      pid_array, cpu_array, data_array ) = embedded_blob
                    t.time_array.extend(time_array)
                    t.line_array.extend(line_array)
                    t.comm_array.extend(comm_array)
                    t.pid_array.extend(pid_array)
                    t.cpu_array.extend(cpu_array)
                    t.data_array.extend(data_array)
        else:
            # There are only two formats, so if we get here
            # then assume the data is sent as a dataframe.
            for t in self.trace_classes:
                if t.name == trace_class_name:
                    t.data_frame = pd.concat([t.data_frame, embedded_blob])

    # this function is used in both single and multiprocess mode.
    # In single process mode, the trace_data member is a single value
    # which can be any iterable type. This method is the one which
    # decides which implementation to use.
    # When called in the dispatcher in multiprocess mode, trace_data is
    # an iterable value rather than a tuple, and since we have
    # self.multiprocess_count > 1, we send the data via
    # self.__parse_trace_queue_impl.
    # As data packets (arrays of trace data lines) arrive at the workers
    # they are sent through this function as tuples consisting of the offset
    # into the file together with the array of lines. These are parsed by
    # the file implementation.
    def __parse_trace_data(self, trace_data, window, abs_window):
        """parse the trace and create a pandas DataFrame"""

        cls_for_unique_word = self.__memoize_unique_words()
        if len(cls_for_unique_word) == 0:
            return

        # when we are multiprocessing, each obj is passed a tuple with an
        # offset and data
        try:
            line_prefix, lines = trace_data
            self.__parse_trace_file_impl(lines, cls_for_unique_word, window, abs_window, line_prefix)
        except ValueError:
            # single process, just do inline on file/list
            if self.multiprocess_count <= 1:
                self.__parse_trace_file_impl(trace_data, cls_for_unique_word,
                                             window, abs_window)
            else:
                self.__parse_trace_queue_impl(trace_data, cls_for_unique_word,
                                              window, abs_window)
    # this function is called by a worker to push it's parsed
    # data onto the output queue, so that the dispatcher can get
    # it back.
    def __push_to_output_q(self, output_q):
        # set use_dataframes=True to generate dataframes in the worker and send
        # set it False to skip generating dataframes and send arrays instead.
        use_dataframes = True
        returned_data = None

        # If we are using dataframes, we need to generate them first.
        if use_dataframes:
            self.finalize_objects()

        # Check each trace class for data and put it on the queue if there is any
        for trace_class in self.trace_classes:
            if use_dataframes:
                if not trace_class.data_frame.empty:
                    returned_data = trace_class.data_frame
            else:
                if len(trace_class.data_array) > 0:
                    returned_data = ( trace_class.time_array,
                                      trace_class.line_array,
                                      trace_class.comm_array,
                                      trace_class.pid_array,
                                      trace_class.cpu_array,
                                      trace_class.data_array )
            # returned_data will be populated with either a dataframe or
            # a tuple with all the arrays embedded in it depending upon
            # configuration above.
            # If the trace class has no data, returned_data will be None.
            if returned_data is not None:
                item = (trace_class.name, self.line_offset, returned_data)
                output_q.put(item, True)

    # __parse_trace became two versions in the multiprocess version.
    # __parse_trace_queue_impl is the version which implements creating
    # worker processes and sending/receiving data from them all.
    def __parse_trace_queue_impl(self, trace_data, cls_for_unique_word, window,
                                 abs_window):

        # __worker_process is the function which is called as main for the
        # processes in the worker pool. It waits for input on input_q,
        # sends it to be parsed, and then returns the parsed data on
        # output_q. The lifecycle is:
        # blocking read from input_q (0.1s timeout)
        #  if read something, parse it immediately
        #  if nothing is there, count the number of times the queue was empty
        #  when we read something, reset the empty queue counter
        #  If we get nothing enough times, send any objects we have so far
        #  along with an end notification, then exit.
        def __worker_process(input_q, output_q, cls_for_unique_word,
                             window, abs_window, events):
            _obj = None
            fails = 0
            max_fails = 10
            while True:
                # use a new instance of the class for each blob we parse
                try:
                    data = input_q.get(True, 0.1)
                    fails = 0
                    if _obj:
                        _obj.__parse_trace_data(data, window, abs_window)
                    else:
                        _obj = FTrace( data=data, events=events, window=window,
                                   abs_window=abs_window, normalize_time=False)
                except Exception as e:
                    fails += 1
                    if fails > max_fails:
                        if _obj:
                            _obj.__push_to_output_q(output_q)
                        output_q.put(("::worker_end: {}".format(os.getpid()), 0, ""), True)
                        return
                    else:
                        pass

        # create input and output queues
        # (input and output always named from
        #  the perspective of the dispatcher)
        input_q = multiprocessing.Queue()
        output_q = multiprocessing.Queue()

        # prepare input
        try:
            if type(trace_data) is not ListType:
                # assume if not a list we got a filename
                lines = open(trace_data)
            else:
                # If we are passed a list, use that as the iterable type
                # This should never happen... at least it hasn't been tested
                print "queue implementation passed data? "
                lines = trace_data
        except FTraceParseError as e:
            raise ValueError('Failed to parse ftrace data {}:\n{}'.format(
                trace_data, str(e)))

        # Create the worker processes
        # Store references in processes for later housekeeping
        processes = []
        for _ in range(0, self.multiprocess_count):
            p=multiprocessing.Process(target=__worker_process,
                                                     args=(input_q, output_q,
                                                     cls_for_unique_word,
                                                     window, abs_window,
                                                     self.requested_events))
            # daemon allows child processes to be automatically
            # killed when the parent dies.
            p.daemon = True
            # child processes should start running immediately
            p.start()
            processes.append(p)

        # __get_line_range is a data collection utility, it maybe
        # doesn't belong here. What it does is to take the iterable
        # source and return the requested range of lines as an
        # array. If we pass a file in, we need to use start=0 always
        # as reading will increment the position. If we pass a bare
        # array in (which we have not tried yet) then this might not
        # work unless we set up a stream on the array or something
        def __get_line_range(fin, start, end):
            array=[]
            for line in itertools.islice(fin, start, end):
                array.append(line)
            return array

        # item is a counter for dispatched items
        item = 0
        # data_to_send is True while we haven't sent everything out
        data_to_send = True
        # I was not convinced we don't lose a few lines here and there
        # so I added this overlapping thing where I send some extra lines
        # in each block except the last. I think it is unnecessary now.
        overlap = 100
        # next_block_start is the line counter for the first line in the
        # next set we are going to send.
        next_block_start = 0
        while True:
            try:
                # first send some data. At the beginning, all the workers have been
                # created and will only wait 1s(ish) before starting to die.
                if data_to_send:
                    # send out 16 blocks per worker each time until we have exhausted
                    # the input data
                    for _ in range(self.multiprocess_count * 16):
                        send_lines = __get_line_range(lines, 0, self.block_len+overlap)
                        if len(send_lines):
                            # data is not exhausted yet
                            # This is a blocking put. If we have a queue length limit
                            # (say, to reduce memory use) then this will throttle.
                            input_q.put((next_block_start, send_lines), True)
                            next_block_start += len(send_lines)
                            # If we remove the overlap, this bit will need to go too.
                            if len(send_lines) > self.block_len:
                                next_block_start -= overlap
                        else:
                            # if we get a 0-length array or None back
                            # then we reached the end of the input
                            data_to_send = False
                            break
                # next, look for some data back from a worker.
                # ideally, we will be reading from the file at the same time as the workers are
                # parsing the data, in order to get maximum efficiency.
                # This is a blocking read with an 0.1s timeout - it will raise Queue.Empty
                # if there is nothing there.
                df_blob = output_q.get(True, 0.1)
                # first item in a returned data blob is always a string.
                # it can be either the trace event name or a string formatted like
                # "::worker_end:<worker pid>"
                str = df_blob[0]
                if str[:13] != "::worker_end:":
                    # This is data to receive
                    item += 1
                    self.__receive_data_blob(df_blob)
                else:
                    # This is a signal that a worker decided there was nothing left to
                    # do. Lets join it so it can die in peace.
                    pid_dying = int(str[14:])
                    for p in processes:
                        if p.pid == pid_dying:
                            p.join()
            except Exception as e:
                # something caused an exception. This is expected if the queues run dry
                # or fill up, so carry on.
                pass

            # an exception above is the only way we exit the dispatch/recv loop
            # lets try to figure out what happened.
            # If we have any still-living workers, we should go round again
            end = True
            for p in processes:
                if p.is_alive():
                    end = False
            if end:
                # here we have no living workers, so we should turn our data
                # into dataframes ready to use
                # first, allow all the children to go away cleanly
                # this should force them to flush anything in their pipes
                # which is not visible to us yet
                for p in processes:
                    p.join()
                # if there is nothing outstanding from the child processes
                # we can go ahead and generate our own dataframes
                if output_q.qsize() == 0:
                    self.finalize_objects()
                    # we may not have this attribute, depending on which
                    # derived interface we used to get here.
                    if getattr(self, 'should_normalize_time', None):
                        self.normalize_time()
                    break
            # this is debug, I used to send end signals to workers - might still go back to that
            #else:
            #    if not data_to_send:
            #        input_q.put((-1,["",]))

    # __parse_trace became two versions in the multiprocess version.
    # __parse_trace_file_impl is used when we are parsing data from a file
    #   directly, and does the normal trace parsing we are used to.
    # It can probably be changed to only get data via a filename and not
    # via the list format. We should perhaps change this to expect an
    # open stream instead of using a filename passed in as trace_data
    # and then we'd be able to either use a pipe, or an array stream
    # or even a pipe directly from trace-cmd report.
    def __parse_trace_file_impl(self, trace_data, cls_for_unique_word,
                                window, abs_window, first_line=0):
        """trace_data is expected to be one of the following:
           * a filename
           * a list of trace data lines

           If a filename is supplied, we read the lines into a list
           and then parse the list.
        """
        try:
            if type(trace_data) is not ListType:
                with open(trace_data) as fin:
                    lines = fin.readlines()[first_line:]
            else:
                # If we are passed a list, parse it all.
                lines = trace_data

            self.line_offset = first_line
            self.lines = first_line
            self.__populate_data(lines, cls_for_unique_word)
        except FTraceParseError as e:
            raise ValueError('Failed to parse ftrace data {}:\n{}'.format(
                trace_data, str(e)))

    # TODO: Move thermal specific functionality

    def get_all_freqs_data(self, map_label):
        """get an array of tuple of names and DataFrames suitable for the
        allfreqs plot"""

        cpu_in_freqs = self.cpu_in_power.get_all_freqs(map_label)
        cpu_out_freqs = self.cpu_out_power.get_all_freqs(map_label)

        ret = []
        for label in map_label.values():
            in_label = label + "_freq_in"
            out_label = label + "_freq_out"

            cpu_inout_freq_dict = {in_label: cpu_in_freqs[label],
                                   out_label: cpu_out_freqs[label]}
            dfr = pd.DataFrame(cpu_inout_freq_dict).fillna(method="pad")
            ret.append((label, dfr))

        try:
            gpu_freq_in_data = self.devfreq_in_power.get_all_freqs()
            gpu_freq_out_data = self.devfreq_out_power.get_all_freqs()
        except KeyError:
            gpu_freq_in_data = gpu_freq_out_data = None

        if gpu_freq_in_data is not None:
            inout_freq_dict = {"gpu_freq_in": gpu_freq_in_data["freq"],
                               "gpu_freq_out": gpu_freq_out_data["freq"]
                           }
            dfr = pd.DataFrame(inout_freq_dict).fillna(method="pad")
            ret.append(("GPU", dfr))

        return ret

    def plot_freq_hists(self, map_label, ax):
        """Plot histograms for each actor input and output frequency

        ax is an array of axis, one for the input power and one for
        the output power

        """

        in_base_idx = len(ax) / 2

        try:
            devfreq_out_all_freqs = self.devfreq_out_power.get_all_freqs()
            devfreq_in_all_freqs = self.devfreq_in_power.get_all_freqs()
        except KeyError:
            devfreq_out_all_freqs = None
            devfreq_in_all_freqs = None

        out_allfreqs = (self.cpu_out_power.get_all_freqs(map_label),
                        devfreq_out_all_freqs, ax[0:in_base_idx])
        in_allfreqs = (self.cpu_in_power.get_all_freqs(map_label),
                       devfreq_in_all_freqs, ax[in_base_idx:])

        for cpu_allfreqs, devfreq_freqs, axis in (out_allfreqs, in_allfreqs):
            if devfreq_freqs is not None:
                devfreq_freqs.name = "GPU"
                allfreqs = pd.concat([cpu_allfreqs, devfreq_freqs], axis=1)
            else:
                allfreqs = cpu_allfreqs

            allfreqs.fillna(method="pad", inplace=True)
            _plot_freq_hists(allfreqs, "out", axis, self.name)

    def plot_load(self, mapping_label, title="", width=None, height=None,
                  ax=None):
        """plot the load of all the clusters, similar to how compare runs did it

        the mapping_label has to be a dict whose keys are the cluster
        numbers as found in the trace and values are the names that
        will appear in the legend.

        """
        import trappy.plot_utils

        load_data = self.cpu_in_power.get_load_data(mapping_label)
        try:
            gpu_data = pd.DataFrame({"GPU":
                                     self.devfreq_in_power.data_frame["load"]})
            load_data = pd.concat([load_data, gpu_data], axis=1)
        except KeyError:
            pass

        load_data = load_data.fillna(method="pad")
        title = trappy.plot_utils.normalize_title("Utilization", title)

        if not ax:
            ax = trappy.plot_utils.pre_plot_setup(width=width, height=height)

        load_data.plot(ax=ax)

        trappy.plot_utils.post_plot_setup(ax, title=title)

    def plot_normalized_load(self, mapping_label, title="", width=None,
                             height=None, ax=None):
        """plot the normalized load of all the clusters, similar to how compare runs did it

        the mapping_label has to be a dict whose keys are the cluster
        numbers as found in the trace and values are the names that
        will appear in the legend.

        """
        import trappy.plot_utils

        load_data = self.cpu_in_power.get_normalized_load_data(mapping_label)
        if "load" in self.devfreq_in_power.data_frame:
            gpu_dfr = self.devfreq_in_power.data_frame
            gpu_max_freq = max(gpu_dfr["freq"])
            gpu_load = gpu_dfr["load"] * gpu_dfr["freq"] / gpu_max_freq

            gpu_data = pd.DataFrame({"GPU": gpu_load})
            load_data = pd.concat([load_data, gpu_data], axis=1)

        load_data = load_data.fillna(method="pad")
        title = trappy.plot_utils.normalize_title("Normalized Utilization", title)

        if not ax:
            ax = trappy.plot_utils.pre_plot_setup(width=width, height=height)

        load_data.plot(ax=ax)

        trappy.plot_utils.post_plot_setup(ax, title=title)

    def plot_allfreqs(self, map_label, width=None, height=None, ax=None):
        """Do allfreqs plots similar to those of CompareRuns

        if ax is not none, it must be an array of the same size as
        map_label.  Each plot will be done in each of the axis in
        ax

        """
        import trappy.plot_utils

        all_freqs = self.get_all_freqs_data(map_label)

        setup_plot = False
        if ax is None:
            ax = [None] * len(all_freqs)
            setup_plot = True

        for this_ax, (label, dfr) in zip(ax, all_freqs):
            this_title = trappy.plot_utils.normalize_title("allfreqs " + label,
                                                        self.name)

            if setup_plot:
                this_ax = trappy.plot_utils.pre_plot_setup(width=width,
                                                        height=height)

            dfr.plot(ax=this_ax)
            trappy.plot_utils.post_plot_setup(this_ax, title=this_title)

class FTrace(GenericFTrace):
    """A wrapper class that initializes all the classes of a given run

    - The FTrace class can receive the following optional parameters.

    :param path: Path contains the path to the trace file.  If no path is given, it
        uses the current directory by default.  If path is a file, and ends in
        .dat, it's run through "trace-cmd report".  If it doesn't end in
        ".dat", then it must be the output of a trace-cmd report run.  If path
        is a directory that contains a trace.txt, that is assumed to be the
        output of "trace-cmd report".  If path is a directory that doesn't
        have a trace.txt but has a trace.dat, it runs trace-cmd report on the
        trace.dat, saves it in trace.txt and then uses that.

    :param name: is a string describing the trace.

    :param normalize_time: is used to make all traces start from time 0 (the
        default).  If normalize_time is False, the trace times are the same as
        in the trace file.

    :param scope: can be used to limit the parsing done on the trace.  The default
        scope parses all the traces known to trappy.  If scope is thermal, only
        the thermal classes are parsed.  If scope is sched, only the sched
        classes are parsed.

    :param events: A list of strings containing the name of the trace
        events that you want to include in this FTrace object.  The
        string must correspond to the event name (what you would pass
        to "trace-cmd -e", i.e. 4th field in trace.txt)

    :param window: a tuple indicating a time window.  The first
        element in the tuple is the start timestamp and the second one
        the end timestamp.  Timestamps are relative to the first trace
        event that's parsed.  If you want to trace until the end of
        the trace, set the second element to None.  If you want to use
        timestamps extracted from the trace file use "abs_window". The
        window is inclusive: trace events exactly matching the start
        or end timestamps will be included.

    :param abs_window: a tuple indicating an absolute time window.
        This parameter is similar to the "window" one but its values
        represent timestamps that are not normalized, (i.e. the ones
        you find in the trace file). The window is inclusive.


    :type path: str
    :type name: str
    :type normalize_time: bool
    :type scope: str
    :type events: list
    :type window: tuple
    :type abs_window: tuple

    This is a simple example:
    ::

        import trappy
        trappy.FTrace("trace_dir")

    """

    def __init__(self, path=".", name="", input_lines=None, normalize_time=True, scope="all",
                 events=[], window=(0, None), abs_window=(0, None), num_processes=1, block_len=10000):
        self.trace_path = self.__process_path(path)

        # If we provide data, we are expecting to just have it parsed,
        # and we will extract the dataframes later so we shall skip
        # populating the metadata
        if not input_lines: # does this even make sense?
            self.__populate_metadata()

        super(FTrace, self).__init__(name, input_lines, normalize_time, scope, events,
                                     window, abs_window, num_processes, block_len)
        self.raw_events = []
        self.trace_path = self.__process_path(path)
        self.__populate_metadata()
        self._do_parse()

    def __warn_about_txt_trace_files(self, trace_dat, raw_txt, formatted_txt):
        self.__get_raw_event_list()
        warn_text = ( "You appear to be parsing both raw and formatted "
                      "trace files. TRAPpy now uses a unified format. "
                      "If you have the {} file, remove the .txt files "
                      "and try again. If not, you can manually move "
                      "lines with the following events from {} to {} :"
                      ).format(trace_dat, raw_txt, formatted_txt)
        for raw_event in self.raw_events:
            warn_text = warn_text+" \"{}\"".format(raw_event)

        raise RuntimeError(warn_text)

    def __process_path(self, basepath):
        """Process the path and return the path to the trace text file"""

        if os.path.isfile(basepath):
            trace_name = os.path.splitext(basepath)[0]
        else:
            trace_name = os.path.join(basepath, "trace")

        trace_txt = trace_name + ".txt"
        trace_raw_txt = trace_name + ".raw.txt"
        trace_dat = trace_name + ".dat"

        if os.path.isfile(trace_dat):
            # Warn users if raw.txt files are present
            if os.path.isfile(trace_raw_txt):
                self.__warn_about_txt_trace_files(trace_dat, trace_raw_txt, trace_txt)
            # TXT traces must always be generated
            if not os.path.isfile(trace_txt):
                self.__run_trace_cmd_report(trace_dat)
            # TXT traces must match the most recent binary trace
            elif os.path.getmtime(trace_txt) < os.path.getmtime(trace_dat):
                self.__run_trace_cmd_report(trace_dat)

        return trace_txt

    def __get_raw_event_list(self):
        self.raw_events = []
        # Generate list of events which need to be parsed in raw format
        for event_class in (self.thermal_classes, self.sched_classes, self.dynamic_classes):
            for trace_class in event_class.itervalues():
                raw = getattr(trace_class, 'parse_raw', None)
                if raw:
                    name = getattr(trace_class, 'name', None)
                    if name:
                        self.raw_events.append(name)

    def __run_trace_cmd_report(self, fname):
        """Run "trace-cmd report [ -r raw_event ]* fname > fname.txt"

        The resulting trace is stored in files with extension ".txt". If
        fname is "my_trace.dat", the trace is stored in "my_trace.txt". The
        contents of the destination file is overwritten if it exists.
        Trace events which require unformatted output (raw_event == True)
        are added to the command line with one '-r <event>' each event and
        trace-cmd then prints those events without formatting.

        """
        from subprocess import check_output

        cmd = ["trace-cmd", "report"]

        if not os.path.isfile(fname):
            raise IOError("No such file or directory: {}".format(fname))

        trace_output = os.path.splitext(fname)[0] + ".txt"
        # Ask for the raw event list and request them unformatted
        self.__get_raw_event_list()
        for raw_event in self.raw_events:
            cmd.extend([ '-r', raw_event ])

        cmd.append(fname)

        with open(os.devnull) as devnull:
            try:
                out = check_output(cmd, stderr=devnull)
            except OSError as exc:
                if exc.errno == 2 and not exc.filename:
                    raise OSError(2, "trace-cmd not found in PATH, is it installed?")
                else:
                    raise
        with open(trace_output, "w") as fout:
            fout.write(out)


    def __populate_metadata(self):
        """Populates trace metadata"""

        # Meta Data as expected to be found in the parsed trace header
        metadata_keys = ["version", "cpus"]

        for key in metadata_keys:
            setattr(self, "_" + key, None)

        with open(self.trace_path) as fin:
            for line in fin:
                if not metadata_keys:
                    return

                metadata_pattern = r"^\b(" + "|".join(metadata_keys) + \
                                   r")\b\s*=\s*([0-9]+)"
                match = re.search(metadata_pattern, line)
                if match:
                    setattr(self, "_" + match.group(1), match.group(2))
                    metadata_keys.remove(match.group(1))

                if SPECIAL_FIELDS_RE.match(line):
                    # Reached a valid trace line, abort metadata population
                    return
