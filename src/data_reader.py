# import
import os
import sys
import json
import math as mh
import numpy as np
import random
import itertools
import pdb
import _pickle as pickle
import scipy.spatial.distance as dist
import scipy.signal as sig
import struct
import copy
import PIL
import io
import time
import copy
import os.path
import argparse
import glob

# class for reading .seq video
FRAME_FORMAT_RAW_GRAY = 100
FRAME_FORMAT_RAW_COLOR = 200
FRAME_FORMAT_JPEG_GRAY = 102
FRAME_FORMAT_JPEG_COLOR = 201

'''
From the MARS Dataset
'''

def load_text_file_as_list(text_file_name):
    """
    Loads a text file as a list of strings.
    """
    file = open(text_file_name, "r")
    text_list = file.readlines()
    string_list = []
    for line in text_list:
        string_list.append(line.strip())
    return string_list


def get_pose_file_name(directory):
    """
    Checks if there is a pose json file in the directory.
    """
    if glob.glob(directory + '/*.json'):
        return glob.glob(directory + '/*.json')
    return False


def get_annotation_file_name(directory):
    """
    Checks if there is an annotation file in the directory.
    """
    if glob.glob(directory + '/*.txt'):
        return glob.glob(directory + '/*.txt')

    if glob.glob(directory + '/*.annot'):
        return glob.glob(directory + '/*.annot')
    return False


def get_sequence_file_name(directory):
    """
    Checks if there is a sequence file in the directory.
    """
    if glob.glob(directory + '/*.seq'):
        return glob.glob(directory + '/*.seq')
    return False


def load_annotations_from_directory_list(directory_list):
    """
    Loads all annotations from input directory list as a dictionary.
    """
    annotation_list = {}

    for directory in directory_list:
        annotation_file_name = get_annotation_file_name(directory)

        if annotation_file_name:
            # Parse the first annotation file in the list.
            annotations = parse_annotations(
                annotation_file_name[0], use_channels=["Ch1"])
            annotations = annotations['behs_frame']
            annotation_list[directory] = annotations

    return annotation_list


def load_annotations_from_file_list(directory_list):
    """
    Loads all annotations from input directory list as a dictionary.
    """
    annotation_list = []

    for annotation_file_name in directory_list:
        #annotation_file_name = get_annotation_file_name(directory)

        if annotation_file_name:
            # Parse the first annotation file in the list.
            annotations = parse_annotations(
                annotation_file_name, use_channels=["S1"])
            annotations = annotations['behs_frame']
            annotation_list.append(annotations)

    return annotation_list


def load_pose_from_directory_list(directory_list):
    """
    Loads all poses from input directory list as a dictionary.
    """
    pose_list = {}
    for directory in directory_list:
        pose_file_name = get_pose_file_name(directory)

        if pose_file_name:
            # Parse the first pose file in the list.
            keypoints_resident, keypoints_intruder = load_pose_keypoints(
                pose_file_name[0])
            pose_list[directory] = np.stack(
                (keypoints_resident, keypoints_intruder), axis=0)

    return pose_list


def load_pose_from_file_list(file_list):
    """
    Loads all poses from input directory list as a dictionary.
    """
    pose_list = {}
    for file in file_list:
        #pose_file_name = get_pose_file_name(directory)
        file = file.split('|')[0]
        if file:
            # Parse the first pose file in the list.
            keypoints_resident, keypoints_intruder = load_pose_keypoints(
                file)
            pose_list[file] = np.stack(
                (keypoints_resident, keypoints_intruder), axis=0)
            print(file)

    return pose_list


def load_pose_and_annotations_from_file_list(file_list):
    """
    Loads all poses from input directory list as a dictionary.
    """
    pose_and_annotation_list = {}
    for file in file_list:
        #pose_file_name = get_pose_file_name(directory)

        if file:
            # Parse the first pose file in the list.
            pose_file = file.split('|')[0]
            annotation_file = file.split('|')[1]

            keypoints_resident, keypoints_intruder = load_pose_keypoints(
                pose_file)
            pose_keypoints = np.stack(
                (keypoints_resident, keypoints_intruder), axis=0)

            annotations = parse_annotations(annotation_file, use_channels = ["Ch1"])
            annotations = annotations['behs_frame']

            pose_and_annotation_list[file] = {"pose": pose_keypoints,
                                              "annotations": annotations}

            print(set(annotations))

    return pose_and_annotation_list


def load_features_and_annotations_from_file_list(file_list):
    """
    Loads all poses from input directory list as a dictionary.
    """
    pose_and_annotation_list = {}
    for i, file in enumerate(file_list):
        #pose_file_name = get_pose_file_name(directory)

        if file:
            # Parse the first pose file in the list.
            pose_file = file.split('|')[2]
            annotation_file = file.split('|')[1]

            print(i, pose_file)
            #try:
            #    end = 'v1_8.npz'
            #    features = np.load(pose_file.split(end)[0] + 'simple_v1_8.npz')
            #except FileNotFoundError:
            #    end = 'v1_7.npz'
            #    features = np.load(pose_file.split(end)[0] + 'simple_v1_7.npz')
            features = np.load(pose_file)
            new_features = features["data_smooth"]
            new_features = np.concatenate([new_features[0, :, :], new_features[1, :, :]], axis = 1)
            print(new_features.shape)
            
            # Remove features that are the same.
            #new_features = np.unique(new_features, axis=1)
            #print(new_features[0, :])

            annotations = parse_annotations(
                annotation_file, use_channels=["Ch1"])
            annotations = annotations['behs_frame']

            print(new_features.shape, len(annotations))

            pose_and_annotation_list[file] = {"features": new_features,
                                              "annotations": annotations}

            print(set(annotations))

    return pose_and_annotation_list


def load_win_features_and_annotations_from_file_list(file_list):
    """
    Loads all poses from input directory list as a dictionary.
    """
    pose_and_annotation_list = {}
    for file in file_list:
        #pose_file_name = get_pose_file_name(directory)

        if file:
            # Parse the first pose file in the list.
            pose_file = file.split('|')[2]
            annotation_file = file.split('|')[1]

            version_names = pose_file[-8:-4]
            pose_file = pose_file.split('_top_')[0] + "_top_simple_" +version_names + "_wnd.npz"
            features = np.load(pose_file)
            #print(features["data"].shape)
            new_features = features["data"]
            #new_features = np.concatenate([new_features[0, :, :], new_features[1, :, :]], axis = 1)
            print(new_features.shape)
            
            # Remove features that are the same.
            #new_features = np.unique(new_features, axis=1)
            #print(new_features[0, :])

            annotations = parse_annotations(
                annotation_file, use_channels=["Ch1"])
            annotations = annotations['behs_frame']

            print(new_features.shape, len(annotations))

            pose_and_annotation_list[file] = {"features": new_features,
                                              "annotations": annotations}

            print(set(annotations))

    return pose_and_annotation_list    


def convert_features_annotation_dictionary_to_list(dictionary):
    feature_list = []
    annotation_list = []

    for key in dictionary.keys():
        # if len(feature_list) == 0:
        #     feature_list = dictionary[key]["features"]
        # else:
        #     feature_list = np.concatenate([feature_list, dictionary[key]["features"]], axis = 0)

        # if len(annotation_list) == 0:
        #     annotation_list = dictionary[key]["annotations"]
        # else:
        #     annotation_list = np.concatenate([annotation_list, dictionary[key]["annotations"]], axis = 0)

        feature_list.append(dictionary[key]["features"])
        annotation_list.append(dictionary[key]["annotations"])
    return feature_list, annotation_list


def convert_pose_annotation_dictionary_to_list(dictionary):
    pose_list = []
    annotation_list = []

    for key in dictionary.keys():
        # if len(feature_list) == 0:
        #     feature_list = dictionary[key]["features"]
        # else:
        #     feature_list = np.concatenate([feature_list, dictionary[key]["features"]], axis = 0)

        # if len(annotation_list) == 0:
        #     annotation_list = dictionary[key]["annotations"]
        # else:
        #     annotation_list = np.concatenate([annotation_list, dictionary[key]["annotations"]], axis = 0)

        pose_list.append(dictionary[key]["pose"].transpose((1,0,2,3)))
        annotation_list.append(dictionary[key]["annotations"])
    return pose_list, annotation_list




def load_pose_keypoints(file_name):
    with open(file_name) as f:
        data = json.load(f)

    # keypoints is a field stored inside data
    keypoints = np.array(data['keypoints'])

    # Keypoints is stored in format: [frame number x 0/1 x 2 x 7]
    keypoints_resident = keypoints[:, 0, :, :]
    keypoints_intruder = keypoints[:, 1, :, :]

    return keypoints_resident, keypoints_intruder


def get_sequence_reader(filename):
    # Read sequence file
    sr = seq_reader(filename)
    sr.build_seek_table()
    num_frames = sr.header['allocated_frames']
    if len(sr.seek_table) != num_frames:
        sr.timestamp_length = 16
        sr.build_seek_table()
    return sr


class seq_reader():

    def __init__(self, filename):
        """Creates a seq_reader, opens file for reading"""
        self.filename = filename
        self.file = open(filename, "rb")
        self.frames_read = -1
        self.header = {}
        self.seek_table = None
        self.timestamp_length = 10

        self.parse_header()

    def parse_header(self):
        """Parse SEQ header"""
        # Make sure we only do this at the beginning of the file
        assert self.frames_read == -1, "Can only read header from beginning of file"

        # Read 1024 bytes (len of header)
        data = self.file.read(1024)
        self.frames_read += 1

        # Read in the struct data
        self.header['magic_number'], \
            self.header['name'], \
            self.header['version'], \
            self.header['header_size'], \
            self.header['description'], \
            self.header['image_width'], \
            self.header['image_height'], \
            self.header['bit_depth'], \
            self.header['bit_depth_real'], \
            self.header['image_size'], \
            self.header['image_format'], \
            self.header['allocated_frames'], \
            self.header['origin'], \
            self.header['true_image_size'], \
            self.header['frame_rate'], \
            self.header['description_format'], \
            self.header['padding'] = struct.unpack(
                'i24sii512sIIIIIiIIIdi428s', data)

        # Set some convenience properties
        self.image_format = self.header['image_format']
        if self.header['image_format'] == 101:
            self.header['image_format'] = 102
        self.bit_depth = self.header['bit_depth']

        if self.image_format in (FRAME_FORMAT_JPEG_GRAY, FRAME_FORMAT_JPEG_COLOR):
            self.compressed = True
        else:
            self.compressed = False

        # Do header-check
        self.check_header()

    def check_header(self):
        """Do some simple tests to make sure the header appears OK"""
        assert self.header['magic_number'] == int(
            '0xFEED', 0), "Incorrect magic number"
        assert self.header['header_size'] == 1024, "incorrect header size"
        assert self.header['origin'] == 0, "incorrect origin"

        # My code uses a timestamp_length of 10 bytes, old uses 8. Check if not
        # 10
        if self.bit_depth / 8 * (self.header['image_height'] * self.header['image_width']) + self.timestamp_length \
                != self.header['true_image_size']:
            # If not 10, adjust to actual (likely 8) and print message
            self.timestamp_length = int(self.header['true_image_size']
                                        - (self.header['bit_depth'] / 8 * (
                                            self.header['image_height'] * self.header['image_width'])))

            # print("Adjusted timestamp length to match header, new length: {}".format(self.timestamp_length))

    def read_frames(self, num_frames):
        """Reads num_frames frames, of any supported format"""

        # Make sure the seek table has been built
        if self.seek_table is None:
            self.build_seek_table()

        # Get a list of seek data for desired frames
        frames_to_read = self.seek_table[
            self.frames_read: self.frames_read + num_frames]

        parsed_frames = []

        # Parse frames
        for offset, size in frames_to_read:
            parsed_frames.append(read_frame(offset, size))

        return parsed_frames

    # TODO: delete me and port code to read_frame_by_index
    def read_frame(self, offset, size):
        """Read a single frame of any supported format given the size and offset in the file."""

        # Get the frame data
        self.file.seek(offset)
        data = self.file.read(size + self.timestamp_length)

        # Parse the frame
        if self.compressed:
            return self.parse_jpeg_frame(data)
        else:
            return self.parse_raw_frame(data)

    def read_frame_by_index(self, index):
        """Read a single frame of any supported format given the frame index."""

        # Make sure the seek table has been built
        if self.seek_table is None:
            self.build_seek_table()

        offset, size = self.seek_table[index]

        # Get the frame data
        self.file.seek(offset)
        data = self.file.read(size + self.timestamp_length)

        # Parse the frame
        if self.compressed:
            return self.parse_jpeg_frame(data)
        else:
            return self.parse_raw_frame(data)

    def parse_jpeg_frame(self, data):
        """Parse the raw jpeg_formatted frame to a bitmap"""
        im = PIL.Image.open(io.BytesIO(data[:-self.timestamp_length]))
        return (np.array(im), data[-self.timestamp_length:])

    # Parse a raw frame and return the image and a timestamp
    # Currently only works for 8 and 16-bit greyscale
    def parse_raw_frame(self, frame):
        # Is the frame greyscale?
        start = time.clock()
        if self.image_format == FRAME_FORMAT_RAW_GRAY:
            # What is the bit depth?
            if self.bit_depth == 16:
                # Parse frame as a list of unsigned 2-byte shorts
                parsed_frame = struct.unpack("{}H".format(self.header['image_height'] * self.header['image_width']),
                                             frame[:-self.timestamp_length])
            elif self.bit_depth == 8:

                # Parse frame as a list of unsigned 1-byte chars
                parsed_frame = struct.unpack("{}B".format(self.header['image_height'] * self.header['image_width']),
                                             frame[:-self.timestamp_length])

            else:
                raise NotImplementedError("Bit depth not supported")
        # Currently don't support color
        else:
            raise NotImplementedError("Only greyscale supported")

        # Reshape to the proper image shape
        shape = (self.header['image_height'], self.header['image_width'])
        parsed_frame = list(parsed_frame)
        # print(type(parsed_frame[0]))
        # parsed_frame = np.asarray(parsed_frame)

        # This is MUCH faster. Use if you need speed. Breaks python import compatability.
        # parsed_frame = toarr(parsed_frame)

        # print(time.clock() - start)
        parsed_frame = np.asarray(parsed_frame)

        parsed_frame = np.reshape(parsed_frame, shape)
        # parsed_frame = np.reshape(parsed_frame, (self.header['image_height'], self.header['image_width']))

        # Get the timestamp
        timestamp = frame[-self.timestamp_length:]

        # Return
        return parsed_frame, timestamp

    def build_seek_table(self, memoize=False):
        """Build a seek table containing the offset and frame size
        for every frame in the video."""

        pickle_name = self.filename.strip(".seq") + ".pickle"
        if memoize:
            if os.path.isfile(pickle_name):
                self.seek_table = pickle.load(open(pickle_name, 'rb'))
                return

        seek_table = []

        if self.compressed:
            # Seek to beginning of frame data
            self.file.seek(1024, 0)
            while (True):
                try:
                    # Read the size
                    size = self.file.read(4)
                    size_parsed = struct.unpack("i", size)[0]

                    # Get the location of this frame
                    offset = self.file.tell()

                    # Seek to the next frame
                    self.file.seek(size_parsed - 4 + self.timestamp_length, 1)

                    # Store the info
                    seek_table.append((offset, size_parsed))

                except:
                    # Most likely error, and what we want to catch, is EOF
                    break
        else:
            frames = range(0, self.header["allocated_frames"])
            offsets = [x * self.header["true_image_size"] +
                       1024 for x in frames]
            for offset in offsets:
                seek_table.append((offset, self.header["image_size"]))

        self.seek_table = seek_table

        if memoize:
            pickle.dump(seek_table, open(pickle_name, 'wb'))

    # Close the file
    def close(self):
        self.file.close()

# how to use this class
# structure for trajectories
# sr = seq_reader(input_video)
# sr.build_seek_table()
# num_frames = sr.header['allocated_frames']
# if len(sr.seek_table) != num_frames:
#     sr.timestamp_length = 16
#     sr.build_seek_table()
# fps = sr.header['frame_rate']
# frame = sr.read_frame_by_index(f)[0]


def parse_annotations(fid, use_channels=[], timestamps=[]):
    # use this function to load annotations of either file type!
    if fid.endswith('.txt'):
        ann_dict = parse_txt(fid)
        return ann_dict
    elif fid.endswith('.annot'):
        ann_dict = parse_annot(fid, use_channels, timestamps)
        return ann_dict


def parse_txt(f_ann):
    header = 'Caltech Behavior Annotator - Annotation File'
    conf = 'Configuration file:'
    fid = open(f_ann)
    ann = fid.read().splitlines()
    fid.close()
    NFrames = []
    # check the header
    assert ann[0].rstrip() == header
    assert ann[1].rstrip() == ''
    assert ann[2].rstrip() == conf
    # parse action list
    l = 3
    names = [None] * 1000
    keys = [None] * 1000
    k = -1

    # get config keys and names
    while True:
        ann[l] = ann[l].rstrip()
        if not isinstance(ann[l], str) or not ann[l]:
            l += 1
            break
        values = ann[l].split()
        k += 1
        names[k] = values[0]
        keys[k] = values[1]
        l += 1
    names = names[:k + 1]
    keys = keys[:k + 1]

    # read in each stream in turn until end of file
    bnds0 = [None] * 10000
    types0 = [None] * 10000
    actions0 = [None] * 10000
    nStrm1 = 0
    while True:
        ann[l] = ann[l].rstrip()
        nStrm1 += 1
        t = ann[l].split(":")
        l += 1
        ann[l] = ann[l].rstrip()
        assert int(t[0][1]) == nStrm1
        assert ann[l] == '-----------------------------'
        l += 1
        bnds1 = np.ones((10000, 2), dtype=int)
        types1 = np.ones(10000, dtype=int) * -1
        actions1 = [None] * 10000
        k = 0
        # start the annotations
        while True:
            ann[l] = ann[l].rstrip()
            t = ann[l]
            if not isinstance(t, str) or not t:
                l += 1
                break
            t = ann[l].split()
            type = [i for i in range(len(names)) if t[2] == names[i]]
            type = type[0]
            if type == None:
                print('undefined behavior' + t[2])
            if bnds1[k - 1, 1] != int(t[0]) - 1 and k > 0:
                print('%d ~= %d' % (bnds1[k, 1], int(t[0]) - 1))
            bnds1[k, :] = [int(t[0]), int(t[1])]
            types1[k] = type
            actions1[k] = names[type]
            k += 1
            l += 1
            if l == len(ann):
                break
        if nStrm1 == 1:
            nFrames = bnds1[k - 1, 1]
        assert nFrames == bnds1[k - 1, 1]
        bnds0[nStrm1 - 1] = bnds1[:k]
        types0[nStrm1 - 1] = types1[:k]
        actions0[nStrm1 - 1] = actions1[:k]
        if l == len(ann):
            break
        while not ann[l]:
            l += 1

    bnds = bnds0[:nStrm1]
    types = types0[:nStrm1]
    actions = actions0[:nStrm1]

    idx = 0

    if len(actions) > 1:
        if len(actions[0]) < len(actions[1]):
            idx = 1
    type_frame = []
    action_frame = []
    len_bnd = []

    for i in range(len(bnds[idx])):
        numf = bnds[idx][i, 1] - bnds[idx][i, 0] + 1
        len_bnd.append(numf)
        action_frame.extend([actions[idx][i]] * numf)
        type_frame.extend([types[idx][i]] * numf)

    ann_dict = {
        'keys': keys,
        'behs': names,
        'nstrm': nStrm1,
        'nFrames': nFrames,
        'behs_se': bnds,
        'behs_dur': len_bnd,
        'behs_bout': actions,
        'behs_frame': action_frame
    }

    return ann_dict


def parse_annot(filename, use_channels=[], timestamps=[]):
    """ Takes as input a path to a .annot file and returns the frame-wise behavioral labels. Optional input use_channels
    only returns annotations in the specified channel(s); default behavior is to merge all channels. Passing timestamps
    from a seq movie will make sure that annotated times are converted to frame numbers correctly in the instance where
    some frames are dropped."""
    if not filename:
        print("No filename provided")
        return -1

    behaviors = []
    channel_names = []
    keys = []

    channel_dict = {}
    with open(filename, 'r') as annot_file:
        line = annot_file.readline().rstrip()
        # Parse the movie files
        while line != '':
            line = annot_file.readline().rstrip()
            # Get movie files if you want

        # Parse the stim name and other stuff
        line = annot_file.readline().rstrip()
        split_line = line.split()
        stim_name = split_line[-1]

        line = annot_file.readline().rstrip()
        split_line = line.split()
        try:
            start_frame = int(float(split_line[-1]))
        except ValueError:
            print(f"[Error] Cannot parse start_frame from: {split_line}")
            return {}


        line = annot_file.readline().rstrip()
        split_line = line.split()
        end_frame = int(float(split_line[-1]))

        framerate = 30  # provide a default framerate if the annot file doesn't have one
        line = annot_file.readline().rstrip()
        if(not(line == '')):  # newer annot files have a framerate line added
            split_line = line.split()
            framerate = float(split_line[-1])
            line = annot_file.readline().rstrip()
        assert (line == '')

        # Just pass through whitespace
        while line == '':
            line = annot_file.readline().rstrip()

        # At the beginning of list of channels
        assert 'channels' in line
        line = annot_file.readline().rstrip()
        while line != '':
            key = line
            keys.append(key)
            line = annot_file.readline().\
                rstrip()

        # At beginning of list of annotations.
        line = annot_file.readline().rstrip()
        assert 'annotations' in line
        line = annot_file.readline().rstrip()
        while line != '':
            behavior = line
            behaviors.append(behavior)
            line = annot_file.readline().rstrip()

        # At the start of the sequence of channels
        line = annot_file.readline()
        while line != '':
            # Strip the whitespace.
            line = line.rstrip()

            assert ('----------' in line)
            channel_name = line.rstrip('-')
            # sloppy fix for now, to get simplified channel name----------------
            channel_name = channel_name[:3]
            channel_names.append(channel_name)

            behaviors_framewise = [''] * end_frame
            line = annot_file.readline().rstrip()
            while '---' not in line:

                # If we've reached EOF (end-of-file) break out of this loop.
                if line == '':
                    break

                # Now get rid of newlines and trailing spaces.
                line = line.rstrip()

                # If this is a blank
                if line == '':
                    line = annot_file.readline()
                    continue

                # Now we're parsing the behaviors
                if '>' in line:
                    curr_behavior = line[1:]
                    # Skip table headers.
                    annot_file.readline()
                    line = annot_file.readline().rstrip()

                # Split it into the relevant numbers
                start_stop_duration = line.split()

                # Collect the bout info.
                # parse bouts that are in frames
                if all('.' not in s for s in start_stop_duration):
                    bout_start = max(
                        (int(start_stop_duration[0]), start_frame - 1))
                    bout_end = min(
                        (int(start_stop_duration[1]), end_frame - start_frame + 1))
                    bout_duration = int(start_stop_duration[2])
                elif len(timestamps) != 0:
                    bout_start = max((np.where(np.append(timestamps, np.inf) >= float(
                        start_stop_duration[0]))[0][0], start_frame - 1))
                    bout_end = min((np.where(np.append(timestamps, np.inf) >= float(
                        start_stop_duration[1]))[0][0], end_frame - start_frame + 1))
                    bout_duration = bout_end - bout_start
                else:
                    bout_start = max(
                        (int(round(float(start_stop_duration[0]) * framerate)), start_frame - 1))
                    bout_end = min(
                        (int(round(float(start_stop_duration[1]) * framerate)), end_frame - start_frame + 1))
                    bout_duration = bout_end - bout_start

                # Store it in the appropriate place.
                if(bout_start <= end_frame):
                    behaviors_framewise[
                        (bout_start - 1):bout_end] = [curr_behavior] * (bout_duration + 1)

                line = annot_file.readline()

                # end of channel
            channel_dict[channel_name] = behaviors_framewise

        # for now, we'll just merge kept channels together, in order listed. this can cause behaviors happening in
        # earlier channels to be masked by other behaviors in later channels, so down the line we should change this to
        # do a smart-merge based on what behaviors we're looking for
        behFlag = 0
        changed_behavior_list = ['other'] * end_frame
        if not use_channels:
            use_channels = channel_names
        for ch in use_channels:
            if (ch in channel_dict):
                chosen_behavior_list = channel_dict[ch]
                if not(behFlag):
                    changed_behavior_list = [annotated_behavior if annotated_behavior != '' else 'other' for annotated_behavior in
                                             chosen_behavior_list]
                    behFlag = 1
                else:
                    changed_behavior_list = [anno[0] if anno[1] == '' else anno[
                        1] for anno in zip(changed_behavior_list, chosen_behavior_list)]
            else:
                print('Did not find a channel' + ch + 'in file ' + filename)
                exit()

        ann_dict = {
            'keys': keys,
            'behs': behaviors,
            'nstrm': len(channel_names),
            'nFrames': end_frame,
            'behs_frame': changed_behavior_list
        }
        return ann_dict


def rast_to_bouts(oneHot, names):  # a helper for the bento save format
    bouts = dict.fromkeys(names, None)
    for val, name in enumerate(names):
        bouts[name] = {'start': [], 'stop': []}
        rast = [annot == val + 1 for annot in oneHot]
        rast = [False] + rast + [False]
        start = [i + 1 for i,
                 (a, b) in enumerate(zip(rast[1:], rast[:-1])) if (a and not b)]
        stop = [i for i, (a, b) in enumerate(
            zip(rast[:-1], rast[1:])) if (a and not b)]
        bouts[name]['start'] = start
        bouts[name]['stop'] = stop
    return bouts


def dump_labels_bento(labels, filename, moviename='', framerate=30, beh_list=['mount', 'attack', 'sniff'], gt=None):

    # Convert labels to behavior bouts
    bouts = rast_to_bouts(labels, beh_list)
    # Open the file you want to write to.
    fp = open(filename, 'wb')
    ch_list = ['classifier_output']
    if gt is not None:
        ch_list.append('ground_truth')
        gt_bouts = rast_to_bouts(gt, beh_list)

    #####################################################

    # Write the header.
    fp.write('Bento annotation file\n')
    fp.write('Movie file(s):\n{}\n\n'.format(moviename))
    fp.write('Stimulus name:\n')
    fp.write('Annotation start frame: 1\n')
    fp.write('Annotation stop frame: {}\n'.format(len(labels)))
    fp.write('Annotation framerate: {}\n\n'.format(framerate))

    fp.write('List of channels:\n')
    fp.write('\n'.join(ch_list))
    fp.write('\n\n')

    fp.write('List of annotations:\n')
    fp.write('\n'.join(beh_list))
    fp.write('\n\n')

    #####################################################
    fp.write('{}----------\n'.format(ch_list[0]))
    for beh in beh_list:
        if beh in bouts.keys():
            fp.write('>{}\n'.format(beh))
            fp.write('Start\tStop\tDuration\n')
            for start, stop in zip(bouts[beh]['start'], bouts[beh]['stop']):
                fp.write('{}\t{}\t{}\t\n'.format(start, stop, stop - start + 1))
            fp.write('\n')
    fp.write('\n')

    if gt is not None:
        fp.write('{}----------\n'.format(ch_list[1]))
        for beh in beh_list:
            if beh in gt_bouts.keys():
                fp.write('>{}\n'.format(beh))
                fp.write('Start\tStop\tDuration\n')
                for start, stop in zip(gt_bouts[beh]['start'], gt_bouts[beh]['stop']):
                    fp.write('{}\t{}\t{}\t\n'.format(
                        start, stop, stop - start + 1))
                fp.write('\n')
        fp.write('\n')

    fp.close()
    return


# convert text labels to id
label2id = {'other': 0,
            'cable_fix': 0,
            'intromission': 0,
            'intruder_introduction': 0,
            'corner': 0,
            'ignore': 0,

            'groom': 0,
            'groom_genital': 0,
            'grooming': 0,
            'socialgrooming': 0,
            'tailrattle': 0,
            'tail_rattle': 0,
            'tailrattling': 0,

            'closeinvestigate': 1,
            'closeinvestigation': 1,
            'investigation': 1,
            'sniff_genitals': 1,
            'sniffurogenital': 1,
            'sniffgenitals': 1,
            'agg_investigation': 1,
            'approach': 1,
            'sniff_face': 1,
            'anogen-investigation': 1,
            'head-investigation': 1,
            'sniff_body': 1,
            'body-investigation': 1,

            'mount': 2,
            'aggressivemount': 2,
            'mount_attempt': 2,

            'attack': 3,
            'intruder_attacks': 3,

            }
