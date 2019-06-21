#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:42:55 2018

@author: Tu Bui tb00083@surrey.ac.uk
"""

import numpy as np
import math
import random
import sys,os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import datetime
import subprocess
import shlex
import skvideo.io as skvid
from skimage.transform import resize as skresize
import cv2
def silent_error_handler(status, func_name, err_msg, file_name, line):
  pass
cv2.redirectError(silent_error_handler) #None if default

def get_opencv_nframes_duration(path, round_down = False):
  """
  using opencv to return frame number and duration, slow but accurate
  :param path: path to video
  :param round_down: round the duration down to integer?
  :return: frame number as integer,duration in seconds as float
  """
  vid = cv2.VideoCapture(path)
  counter = 0
  while vid.grab():
    counter += 1
  d = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.
  d = math.floor(d) if round_down else d
  vid.release()
  return counter, d

class VideoData2(object):
  """
  abstract class reading video data using scikit-video (pip install scikit-video).
  slow, but can correctly retrieve all frames.
  """
  def __init__(self, video_path = None, output_size = None, stride_in_sec = None, video_reader = 'skvideo'):
    encoder_lib = ['skvideo', 'opencv']
    self.cap = None
    self.out_shape = output_size
    self.stride_in_sec = stride_in_sec
    assert video_reader in encoder_lib, 'Error! Video reader only supports {} atm'.format(encoder_lib)
    self.vreader = encoder_lib.index(video_reader)
    if video_path is not None:
      self.setup(video_path)
  
  def setup(self, video_path):
    assert os.path.exists(video_path), 'Error! Path {} doesnot exist'.format(video_path)
    self.path = video_path
    if self.vreader == 0: #skvideo
      if self.cap is not None:
        self.vid.close()
        self.cap.close()
      self.cap = skvid.FFmpegReader(video_path)
      self.vid = self.cap.nextFrame() #a generator
      #self.frame_rate = float(self.cap.probeInfo['video']['@avg_frame_rate'].split('/')[0])
      self.nframes = self.cap._probCountFrames()
      self.duration = self._get_duration(self.cap, round_up = False)
      #self.frame_rate = self.nframes / self.duration #obsolete
    elif self.vreader == 1: #opencv
      if self.cap is not None:
        self.cap.release()
      self.nframes, self.duration = get_opencv_nframes(video_path, True)
      self.duration_method = 'opencv_manual_loop'
      self.cap = cv2.VideoCapture(video_path)

    self.nframes_eff = int(self.duration / self.stride_in_sec) if self.stride_in_sec else self.nframes
    #self.fsteps = self.frame_rate*self.stride_in_sec if self.stride_in_sec else 1 #frame step as float, obsolete
    self.fid = 0
    self.valid_fid = np.linspace(0, self.nframes, self.nframes_eff, endpoint = False).astype(int).tolist()

  def _get_duration_from_meta(self, path):
    """try to extract video length from meta data"""
    cmd = 'ffmpeg -i {} -f ffmetadata -'.format(path)
    args = shlex.split(cmd)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    if exitcode: return None
    if 'Duration:' not in err: return None
    duration = err.split('Duration:')[1].split(',')[0].strip()
    return self._string2time(duration)

  def _string2time(self, timestring):
    """convert time string into time in second
    timestring format: hh:mm:ss.mili"""
    out = timestring.split(':')
    out = [float(t) for t in out]
    out = 3600*out[0] + 60*out[1] + out[2]
    return out

  def _get_duration(self, cap, round_up = True):
    """get duration info in seconds given video FFmpegReader object
    round_up: if True remove the fraction of second"""
    meta = cap.probeInfo['video']
    if '@duration' in meta.keys():
      out = float(meta['@duration'])
      if round_up: out = int(out)
      self.duration_method = 'meta'
      return float(out)

    def find_duration_in_tag(mytag):
      """try to find duration in tag meta"""
      if type(mytag) is list:
        for tag_ in mytag:
          out_ = find_duration_in_tag(tag_)
          if out_ is not None:
            return out_
      elif mytag['@key'] == 'DURATION':
        out_ = self._string2time(mytag['@value'])
        return out_
      else: return None

    if 'tag' in meta.keys():
      out = find_duration_in_tag(meta['tag'])
      if out is not None:
          if round_up: out = int(out)
          self.duration_method = 'tag'
          return float(out)
    #read from meta with ffmpeg
    out = self._get_duration_from_meta(self.path)
    if out:
      if round_up: out = int(out)
      self.duration_method = 'ffmpeg'
      return float(out)
    #final method
    if '@avg_frame_rate' in meta.keys(): #based on frame rate
      frame_rate = float(meta['@avg_frame_rate'].split('/')[0])
      nframes = cap._probCountFrames()
      out = float(nframes) / frame_rate
      if round_up: out = int(out)
      self.duration_method = 'frame_rate'
      return float(out)
          
  def showInfo(self):
    if self.cap is not None:
      print('Video path: {}'.format(self.path))
      print('Effective Frame number: {}'.format(self.nframes_eff))
      print('Original frame number: {}'.format(self.nframes))
      print('Average Frame rate: {}'.format(self.frame_rate))
      print('Duration: {}s, retrieved via {}'.format(self.duration, self.duration_method))
    else: print('N/A. Video hasnot been setup yet')
  
  def get_frame_number(self):
    """
    get frame number wrt stride_in_sec
    e.g. if the video has 10 sec length and stride_in_sec = 0.5
    so effective frame number is 10/0.5 = 20
    """
    return self.nframes_eff
  
  def _get_frame_number(self):
    """
    get exact number of frames
    """
    return self.nframes
  
  # def _get_next(self, num_frames=1):
  #   """legacy code, should not use"""
  #   for frame_id in range(num_frames):
  #     img = next(self.vid)
  #     if frame_id == 0:
  #       out = np.zeros([num_frames,] + list(img.shape), dtype = np.uint8)
  #     out[frame_id,...] = img[:,:,::-1] #BGR to RGB
  #   return out
  
  def get(self, start_id, end_id=None):
    """return 4-D blob of frames
    note: start_id and end_id are effective frame ids"""
    if end_id is None: end_id = start_id + 1
    real_ids = self.valid_fid[start_id:end_id]
    
    if self.fid > real_ids[0]: #read pass this point already, need to reset
      self.setup(self.path)
    imgs =[]
    for i in range(self.fid, real_ids[-1]+1):
      if self.vreader == 0:  # skvideo
        img = next(self.vid)
      elif self.vreader == 1: # opencv
        res, img = self.cap.read()
        assert res, 'Error. OpenCV unable to read frame #%d in %s' % (i, self.path)
        img = img[:,:,::-1]
      if i in real_ids:
        imgs.append(img)
    self.fid = real_ids[-1] + 1
    
    if self.out_shape is not None:
      imgs = [skresize(im, self.out_shape, anti_aliasing=True,mode='reflect') for im in imgs]
    return np.uint8(np.array(imgs)*255)
    
  def __del__(self):
    if self.cap is not None:
      if self.vreader == 0:  # skvideo
        self.vid.close()
        self.cap.close()
      elif self.vreader == 1: #opencv
        self.cap.release()
      self.cap = None
