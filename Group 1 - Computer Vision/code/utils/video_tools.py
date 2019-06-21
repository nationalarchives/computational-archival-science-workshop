#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:34:22 2018
data prep
do all preperation steps: chop, feat extract, cluster, augment 
@author: Tu Bui tb00083@surrey.ac.uk
"""
supported_formats = ['mp4','flv','avi','mov','mkv','wmv','3gp']
supported_codecs =     {'mp4': ['h264',],
                        'flv': ['h264',],
                        'avi': ['h264', 'vp8', 'mpeg2video'],
                        'mov': ['h264', 'h263p'],
                        'mkv': ['vp9', 'h263p','vp8','mpeg2video'],
                        'wmv': ['h264',],
                        '3gp': ['h264', 'mpeg4']}
low_quality_codecs = ['h263p', 'mpeg2video']
frame_rates = [24, 25, 30]
avimkv_fr = [24, 25]
flv_fr = [25,]

supported_formats_a = ['mp3', 'aac','wav','ogg']
supported_codecs_a = {'mp3': ['libmp3lame',],
                      'aac': ['aac',],
                      'wav': ['pcm_s16le',],#PCM signed 16bit little endian
                      'ogg': ['libvorbis',]}
supported_sample_rates_a = [32000, 44100, 48000]

### basic import ###
import sys,os
import shutil
import numpy as np
import subprocess
import shlex
import datetime
import time
import math
import pandas as pd
import errno
from sklearn.decomposition import PCA
from pycluster import pycluster
import json
from concurrent import futures
from pydub import AudioSegment
from pydub.silence import split_on_silence
from python_speech_features import mfcc, logfbank
### deep learning import ###
from keras import backend as K
from keras.models import Model
import keras.applications as apps
#from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input#, decode_predictions
import numpy as np
from DataGenerator import VideoData2 as VideoData
#from DataGenerator import AEDataGenerator
import model_def
from helper import float2hex, makeNewDir
from multiprocessing import Pool

def locateExe(exe_name):
  """search and return file location in sys path"""
  for path in sys.path:
    if os.path.isdir(path) and exe_name in os.listdir(path):
      return os.path.join(path, exe_name)

def runExternal(cmd, return_output = True, env = None):
  """
  Execute the external command and get its exitcode, stdout and stderr.
  """
  args = shlex.split(cmd)
  if return_output:
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env = env)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err
  else:
    return subprocess.check_output(args, stderr=subprocess.STDOUT, env = env)

def hasAudio(video_file):
  """
  check if a video file has audio or not
  """
  cmd = 'ffprobe -i "%s" -show_streams -select_streams a -loglevel error' % video_file
  exitcode, out, err = runExternal(cmd, True)
  assert exitcode == 0, 'Error! External cmd %s failed with code %d - %s' % (cmd, exitcode, err)
  if out: return True
  else: return False

def extractAudioFeat(audio_file, winlen = 1.0, winstep = None, ftype = 'mfcc'):
  """
  read audio file, extract mfcc feature
  window_len: in seconds
  ftype: either mfcc or logfbank or both
  """
  ext = audio_file.split('.')[1]
  au = AudioSegment.from_file(audio_file, ext)
  fr = au.frame_rate
  data = np.frombuffer(au._data, np.int16)
  step = winlen if winstep is None else winstep
  if ftype == 'mfcc':
    out = mfcc(data,fr, winlen=winlen,winstep=step,nfft = int(fr*winlen))
  elif ftype == 'logfbank':
    out = logfbank(data,fr, winlen=winlen,winstep=step,nfft = int(fr*winlen))
  elif ftype == 'both':
    mfcc_feat = mfcc(data,fr, winlen=winlen,winstep=step,nfft = int(fr*winlen))
    fbank_feat = logfbank(data,fr, winlen=winlen,winstep=step,nfft = int(fr*winlen))
    out = np.c_[mfcc_feat, fbank_feat]
  #out = np.ravel(fbank_feat) #np.ravel(mfcc_feat)
  return out

def extract_audio_feat2file(audio_file, output_file, winlen=1.0, winstep = None, ftype = 'mfcc'):
  """extract audio feature and save as numpy array"""
  if os.path.exists(output_file): return 0
  try:
    res = extractAudioFeat(audio_file, winlen, winstep, ftype)
    np.save(output_file, res)
  except Exception as e:
    print('Error happened extracting audio feat at %s: %s' % (audio_file,e))
    return 1
  return 0

def getLength(filename):
  """get length of a video in seconds"""
  result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  vlen =  [x for x in result.stdout.readlines() if "Duration" in x]
  pattern = 'Duration: '
  vlen = vlen[0].split(',')[0]
  vlen = vlen[vlen.find(pattern)+len(pattern):]
  x = time.strptime(vlen.split('.')[0],'%H:%M:%S')
  out = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
  return out

def genTrainList(cluster_list, feat_list, out_dir):
  """generate a list of training file, each responsible to train a model
  also generate a train_info.txt containing overall info"""
  cluster_ = pd.read_csv(cluster_list) #for original clip id and cluster id
  feat_ = pd.read_csv(feat_list) #extracted feat for aug clips, input to training
  makeNewDir(out_dir, False)
  aux = []
  allclusters = list(set(cluster_['cluster_id']))
  allclusters = [i for i in allclusters if i!=-1]
  for c in allclusters:
    clip_ids = [i for i in range(len(cluster_)) if cluster_['cluster_id'][i] == c]
    ids_ = [i for i in range(len(feat_)) if feat_['clip_id'][i] in clip_ids]
    labels = [clip_ids.index(feat_['clip_id'][i]) for i in ids_]
    feat_names = [feat_['path'][i] for i in ids_]
    min_seq = [feat_['min_seq'][i] for i in ids_]
    out_ = pd.DataFrame(columns=['path','label','length'])
    out_['path'] = feat_names
    out_['label'] = labels
    out_['length'] = min_seq
    train_list = os.path.join(out_dir, 'train_list' + str(c) + '.txt')
    out_.to_csv(train_list, index=False)
    aux.append(dict(cluster_id = c, clip_ids = clip_ids, train_list = train_list))

  with open(os.path.join(out_dir, 'train_info.txt'),'w') as fout:
    json.dump(aux, fout, indent=4)
  return aux
  

def genTrainList_audio(feat_list, out_dir):
  """
  simplified version of genTrainList for audio
  """
  makeNewDir(out_dir, False)
  list_a = pd.read_csv(feat_list)
  out_ = pd.DataFrame(columns=['path', 'label', 'length'])
  out_['path'] = list_a['path'].tolist()
  labels = list_a['clip_id'].tolist()
  out_['label'] = labels
  out_['length'] = list_a['min_seq'].tolist()
  c = 0 #there is only one training list for audio
  train_list = os.path.join(out_dir, 'train_list' + str(c) + '.txt')
  out_.to_csv(train_list, index = False)
  aux = []
  aux.append(dict(cluster_id = c, clip_ids = list(set(labels)), train_list = train_list))
  with open(os.path.join(out_dir, 'train_info.txt'),'w') as fout:
    json.dump(aux, fout, indent=4)
  return aux

# =============================================================================
# def videoChop(filename, timestamps, out_dir, include_audio=False, out_dir_audio = None, extra="", **kwargs):
#   """split videos given list of timestamps"""
#   split_cmd = "ffmpeg -i '%s' -y -c:v h264" % (filename)
#   if include_audio:
#     split_cmd += " -c:a wav " #loss less audio
#     out_dir_a = out_dir if out_dir_audio is None else out_dir_audio
#   try:
#     fileext = filename.split(".")[-1]
#   except IndexError as e:
#     raise IndexError("No . in filename. Error: " + str(e))
#   split_points = timestamps
#   count = 0
#   split_list = pd.DataFrame(columns = ['clip_id','start','end','path'])
#   for i in range(len(split_points)-1):
#     split_str = " -filter_complex "
#     split_start = split_points[i] + 0.05
#     split_end = split_points[i+1] - 0.05
#     if split_end - split_start > 1:
#       filebase = 'p' + str(count) + "." + fileext
#       out_path = os.path.join(out_dir,filebase)
#       if include_audio:
#         filebase_a = 'p' + str(count) + ".wav"
#         out_path_a = os.path.join(out_dir_a, filebase_a)
#         split_str += ("'[0:v]trim={}:end={},setpts=PTS-STARTPTS[vout];" 
#                       "[0:a]atrim={}:end={},asetpts=PTS-STARTPTS,pan=mono|c0=.5*c0+.5*c1[aout]'" 
#                       " -map [vout] '{}' -map [aout] '{}'").format(
#             split_start,split_end,split_start, split_end, out_path,out_path_a)
#       else:
#         split_str += "'[0:v]trim={}:end={},setpts=PTS-STARTPTS[vout]' -map [vout] '{}'".format(
#           split_start,split_end,out_path)
#       
#       cmd = split_cmd + split_str
#       print "About to run: "+cmd
#       exitcode,out,err = runExternal(cmd)
#       assert exitcode == 0, 'Error! external command fail.\n{}-{}'.format(out, err)
#       split_list.loc[count] = [count, split_start, split_end, filebase]
#       count += 1
#       
#   return split_list
# =============================================================================

def videoChop(filename, split_points, out_dir, media = 'video'):
  """
  split video to different parts containing video or audio only
  """
  makeNewDir(out_dir, remove_existing = True)
  count = 0
  split_list = pd.DataFrame(columns=['clip_id', 'start', 'end', 'path'])
  fileext = filename.split(".")[-1] if media == 'video' else 'wav'
  if split_points is None:
      filebase = 'p0.' + fileext
      split_list.loc[count] = [0, 0, getLength(filename), filebase]
      shutil.copyfile(filename, os.path.join(out_dir, filebase))
  else:
      if media == 'video':
        split_cmd = ("ffmpeg -i '%s' -y -c:v h264 -filter_complex " 
                     "'[0:v]trim=%.2f:end=%.2f,setpts=PTS-STARTPTS[vout]'"
                     " -map [vout] '%s'")
      else:
        split_cmd = ("ffmpeg -i '%s' -y -c:v h264 -filter_complex " 
                     "'[0:a]atrim=%.2f:end=%.2f,asetpts=PTS-STARTPTS,pan=mono|c0=.5*c0+.5*c1[aout]'"
                     " -map [aout] '%s'")

      for i in range(len(split_points)-1):
        split_start = split_points[i] + 0.05
        split_end = split_points[i+1] - 0.05
        if split_end - split_start > 1:
          filebase = 'p' + str(count) + "." + fileext
          out_path = os.path.join(out_dir,filebase)
          vals = (filename, split_start, split_end, out_path)
          cmd = split_cmd % vals
          print("About to run: "+cmd)
          exitcode,out,err = runExternal(cmd)
          assert exitcode == 0, 'Error! external command fail.\n{}-{}'.format(out, err)
          split_list.loc[count] = [count, split_start, split_end, filebase]
          count += 1
  split_list.to_csv(os.path.join(out_dir, 'list.txt'), index = False, float_format='%.2f')

def audioChop(filename, split_points, out_dir, silence_removal = True):
    """chop audio, normalise and remove silence part"""
    makeNewDir(out_dir, remove_existing=True)
    count = 0
    count_valid = 0
    split_list = pd.DataFrame(columns=['clip_id', 'start', 'end', 'path'])

    if split_points is None:
        filebase = 'p0.wav'
        out_split = os.path.join(out_dir, filebase)
        split_list.loc[count] = [0, 0, getLength(filename), filebase]
        cmd = 'ffmpeg -y -i %s %s' % (filename, out_split)
        exitcode, out, err = runExternal(cmd)
        assert exitcode == 0, 'Error! external command fail: {}.\n{}-{}'.format(cmd, out, err)
        # preprocess
        au = AudioSegment.from_file(out_split, format='wav')
        au = audioPreprocess(au, -20, silence_removal)
        if au.duration_seconds > 1:
            au.export(out_split, format='wav')
    else:
        split_cmd = ("ffmpeg -i '%s' -y -vn -filter_complex "
                     "'[0:a]atrim=%.2f:end=%.2f,asetpts=PTS-STARTPTS,pan=mono|c0=.5*c0+.5*c1[aout]'"
                     " -map [aout] '%s'")

        for i in range(len(split_points) - 1):
            split_start = split_points[i] + 0.05
            split_end = split_points[i + 1] - 0.05
            if split_end - split_start > 1:
                #split
                filebase = 'p' + str(count) + ".wav"
                out_split = os.path.join(out_dir, filebase)
                cmd = split_cmd % (filename, split_start, split_end, out_split)
                exitcode, out, err = runExternal(cmd)
                assert exitcode == 0, 'Error! external command fail: {}.\n{}-{}'.format(cmd, out, err)
                #preprocess
                au = AudioSegment.from_file(out_split, format = 'wav')
                au = audioPreprocess(au, -20, silence_removal)
                if au.duration_seconds > 1:
                    au.export(out_split, format = 'wav')
                    split_list.loc[count_valid] = [count, split_start, split_end, filebase]
                    count_valid += 1
                count += 1

    split_list.to_csv(os.path.join(out_dir, 'list.txt'), index=False, float_format='%.2f')

def audioPreprocess(sound, norm = -20, silence_removal = True):
    """preprocess audio
    IN: sound   AudioSegment object
        norm    rms normalisation (in dB), 0 if turn off
        silence_removal whether to remove silence in audio
    OUT: return a preprocessed AudioSegment object"""
    out = sound
    if norm:
        out = out.apply_gain(norm - out.dBFS)
    if silence_removal:
        #out = out.strip_silence(silence_thresh = -40) doesn't work coz a bug in strip_silence()
        padding = 100
        chunks = split_on_silence(out, 1000, -40, padding)
        crossfade = padding / 2
        if not len(chunks):
            out = out[0:0]
        else:
            out = chunks[0]
            for chunk in chunks[1:]:
                out = out.append(chunk, crossfade=crossfade)
    return out

def getSplitPoints(file_name, out_dir):
  makeNewDir(out_dir, remove_existing = False)
  cmd = locateExe('scene_filter.sh') + " '" + file_name + "' " + " '" + out_dir + "'"
  #print(cmd)
  status,res,err = runExternal(cmd)
  assert status==0,'Opps. ffmpeg error running {}: {}'.format(cmd, err)
  if res.strip() == '-1':
      return None
  print('Split points: \n{}'.format(res))
  res = res.split('\n')
  res = [round(float(x),2) for x in res if x]
  splits = [0,] + res + [getLength(file_name),]
  return splits

def autoVideoChop(file_name, out_dir):
  """split a video into multiple continuous scenes (clips) using ffmpeg
  out_dir will be created during execution
  deprecated: use videoChop and audioChop instead
  """
  print('\n\nautoVideoChop start.')
  video_dir = os.path.join(out_dir, 'video','chop')
  makeNewDir(video_dir, remove_existing = True)
  #check if video has audio
  has_audio = hasAudio(file_name)
  if has_audio:
    audio_dir = os.path.join(out_dir, 'audio','chop')
    makeNewDir(audio_dir, remove_existing = True)
  cmd = locateExe('scene_filter.sh') + " '" + file_name + "' " + " '" + out_dir + "'"
  status,res,err = runExternal(cmd)
  assert status==0,'Opps. ffmpeg error: {}'.format(err)
  print('Split points: \n{}'.format(res))
  res = res.split('\n')
  res = [round(float(x),2) for x in res if x]
  splits = [0,] + res + [getLength(file_name),]
  
  split_points = videoChop(file_name, splits, video_dir,
                           include_audio = has_audio, out_dir_audio = audio_dir,
                           extra = '-threads 4')
  split_points.to_csv(os.path.join(video_dir, 'list.txt'), index = False)
  if has_audio:
    paths = split_points['path'].tolist()
    paths = [p.split('.')[0] + '.wav' for p in paths]
    split_points['path'] = paths
    split_points.to_csv(os.path.join(audio_dir, 'list.txt'), index = False)
  print('Number of clips: {}'.format(split_points.shape[0]))
  print('autoVideoChop ends.\n\n')
  return split_points

def AVConvert(in_file, out_file, vq = 23, vr = 25, vc = 'h264', aq=1, ar=48000, ac='wav', verbose = True):
  """convert both video and audio"""
  cmd = 'avconv -y -i ' + in_file + ' -c:v ' + vc + ' -crf ' + str(vq) + \
   ' -c:a ' + ac + ' -q:a ' + str(aq) + ' -ar ' + str(ar)
  if out_file.split('.')[-1] != 'flv':
    cmd += (' -r ' + str(vr))
  cmd += (' ' + out_file)
  if verbose:
    print('Running %s' % cmd)
  return runExternal(cmd, False)

def AVConvertWrapper(args):
  in_file, out_file, vq, vr, vc, aq, ar, ac = args
  return AVConvert(in_file, out_file, vq, vr, vc, aq, ar, ac)

def videoConvert(in_file, out_file, quality=0, r=25, codec = 'h264'):
  """
  convert a video using ffmpeg
  """

  if out_file.split('.')[-1] == 'flv': #no framerate setting
    cmd = 'avconv -y -i ' + in_file + ' -c:v ' + codec + ' -an ' + ' -crf ' +\
            str(quality) +  ' ' + out_file
  else:
    cmd = 'avconv -y -i ' + in_file + ' -c:v ' + codec + ' -an ' + ' -crf ' +\
            str(quality) + ' -r ' + str(r) + ' ' + out_file
  #os.system(cmd)
  return runExternal(cmd, False)

def videoConvertWrapper(args):
  """
  wrapper of videoConvert for multiprocessing
  random codecs and framerates
  """
  in_file, out_file, quality, framerate, codec = args
  if os.path.exists(out_file): return 0
  return videoConvert(in_file, out_file, quality, framerate, codec)

def audioConvert(in_file, out_file, quality=1, r=48000, codec = 'wav'):
  """
  convert audio with ffmpeg
  quality 1-5
  """
  cmd = 'ffmpeg -i ' + in_file + ' -c:a ' + codec +\
    ' -q:a ' + str(quality) + ' -ar ' + str(r) + ' -y ' + out_file
  return runExternal(cmd, False)

def audioConvertWrapper(args):
  """
  wrapper of videoConvert for multiprocessing
  random codecs and framerates
  """
  in_file, out_file, quality, r, codec = args
  if os.path.exists(out_file): return 0
  return audioConvert(in_file, out_file, quality, r, codec)

def sceneAugment(input_, output_, quality_range = [0,],
                 format_list = None):
  """
  convert a video or list of videos to different formats with different quality
  input_ : path to a video or .txt containing list of the videos
  output_: output path; or a directory if input is a list of videos
  quality_range: range of encoding quality; lower the better
  vformat: list of video formats; can be overwritten if output_ format is specified
  """
  if format_list is not None:
    format_list = list(format_list)
    format_list = [item.strip('.') for item in format_list]
    format_list = [item for item in format_list if item in supported_formats]
    print('Destination video formats: {}'.format(format_list))
  
  parent_dir = os.path.dirname(input_)
  if input_.endswith('.txt'):
    data = pd.read_csv(input_)
    input_ids = data['clip_id'].tolist()
    input_names = data['path']
    input_paths = [os.path.join(parent_dir, name) for name in input_names]
  elif input_.split('.')[-1] in supported_formats: #single input
    input_paths = [input_,]
    input_ids = [0,]
    
  else:
    raise ValueError, "Input {} not recognised".format(input_)
  input_list = []
  output_list = []
  clip_id = []
  quality_list = []
  framerate_list = []
  codec_list = []
  high_quality = []
  for i,video in enumerate(input_paths):
    vname,ext = os.path.basename(video).split('.')
    for q in quality_range:
      hq = 1
      if format_list is None:
        input_list.append(os.path.join(parent_dir, video))
        output_list.append(os.path.join(output_, vname+'_q'+str(q)+'.'+ext))
        clip_id.append(input_ids[i])
        quality_list.append(q)
        fr = np.random.choice(avimkv_fr) if ext in ['avi','mkv'] else np.random.choice(frame_rates)
        framerate_list.append(fr)
        codec = np.random.choice(supported_codecs[ext])
        codec_list.append(codec)
        hq = 0 if codec in low_quality_codecs else 1
        high_quality.append(hq)
      else:
        for format_ in format_list:
          input_list.append(os.path.join(parent_dir, video))
          output_list.append(os.path.join(output_, vname+'_q'+str(q)+'.'+format_))
          clip_id.append(input_ids[i])
          quality_list.append(q)
          fr = np.random.choice(avimkv_fr) if format_ in ['avi','mkv'] else np.random.choice(frame_rates)
          framerate_list.append(fr)
          codec = np.random.choice(supported_codecs[format_])
          codec_list.append(codec)
          hq = 0 if codec in low_quality_codecs else 1
          high_quality.append(hq)
    
  #double check if there is just one output
  nfiles = len(output_list)
  if nfiles==1 and os.path.basename(output_).split('.')[-1] in supported_formats:
    multi = False
    output_list = [output_]
  else:
    multi = True
    makeNewDir(output_, remove_existing = False) #assume output is a dir
    log = pd.DataFrame(columns=('clip_name','path', 'clip_id'))
    log['clip_name'] = [in_.replace(parent_dir + '/','') for in_ in input_list]
    log['path'] = [out_.replace(output_ + '/','') for out_ in output_list]
    log['clip_id'] = clip_id
    log['high_quality'] = high_quality
  # start augmenting
  with futures.ProcessPoolExecutor(8) as executor:
    executor.map(videoConvertWrapper, zip(input_list, output_list, quality_list, framerate_list, codec_list))
  if multi:
    log['clip_id'] = log['clip_id'].astype(int)
    log['high_quality'] = log['high_quality'].astype(int)
    log.to_csv(os.path.join(output_,'list.txt'), index = False)
  

def audioAugment(input_, output_, quality_range = [1,],
                 format_list = None):
  """
  convert a audio or list of audios to different formats with different quality
  input_ : path to a audio or .txt containing list of the audios
  output_: output path; or a directory if input is a list of audios
  quality_range: range of encoding quality; lower the better
  vformat: list of audio formats; can be overwritten if output_ format is specified
  """
  if format_list is not None:
    format_list = list(format_list)
    format_list = [item.strip('.') for item in format_list]
    format_list = [item for item in format_list if item in supported_formats_a]
    print('Destination audio formats: {}'.format(format_list))
  
  parent_dir = os.path.dirname(input_)
  if input_.endswith('.txt'):
    data = pd.read_csv(input_)
    input_ids = data['clip_id'].tolist()
    input_names = data['path']
    input_paths = [os.path.join(parent_dir, name) for name in input_names]
  elif input_.split('.')[-1] in supported_formats_a: #single input
    input_paths = [input_,]
    input_ids = [0,]
    
  else:
    raise ValueError, "Input {} not recognised".format(input_)
  input_list = []
  output_list = []
  clip_id = []
  quality_list = []
  framerate_list = []
  codec_list = []
  high_quality = []
  for i,audio in enumerate(input_paths):
    vname,ext = os.path.basename(audio).split('.')
    for q in quality_range:
      hq = 1
      if format_list is None:
        input_list.append(os.path.join(parent_dir, audio))
        output_list.append(os.path.join(output_, vname+'_q'+str(q)+'.'+ext))
        clip_id.append(input_ids[i])
        quality_list.append(q)
        fr = np.random.choice(supported_sample_rates_a)
        framerate_list.append(fr)
        codec_list.append(np.random.choice(supported_codecs_a[ext]))
        high_quality.append(hq)
      else:
        for format_ in format_list:
          input_list.append(os.path.join(parent_dir, audio))
          output_list.append(os.path.join(output_, vname+'_q'+str(q)+'.'+format_))
          clip_id.append(input_ids[i])
          quality_list.append(q)
          fr = np.random.choice(supported_sample_rates_a)
          framerate_list.append(fr)
          codec_list.append(np.random.choice(supported_codecs_a[format_]))
          high_quality.append(hq)
    
  #double check if there is just one output
  nfiles = len(output_list)
  if nfiles==1 and os.path.basename(output_).split('.')[-1] in supported_formats_a:
    multi = False
    output_list = [output_]
  else:
    multi = True
    makeNewDir(output_, remove_existing = False) #assume output is a dir
    log = pd.DataFrame(columns=('clip_name','path', 'clip_id'))
    log['clip_name'] = [in_.replace(parent_dir + '/','') for in_ in input_list]
    log['path'] = [out_.replace(output_ + '/','') for out_ in output_list]
    log['clip_id'] = clip_id
    log['high_quality'] = high_quality
  # start augmenting
  with futures.ProcessPoolExecutor(8) as executor:
    executor.map(audioConvertWrapper, zip(input_list, output_list, quality_list, framerate_list, codec_list))
  if multi:
    log['clip_id'] = log['clip_id'].astype(int)
    log['high_quality'] = log['high_quality'].astype(int)
    log.to_csv(os.path.join(output_,'list.txt'), index = False)
  

def sceneCluster(in_file, out_file, nsamples = 100):
  """
  cluster a set of scenes extracted from a video
  in_file: file listing scene features
  out_file: scene with cluster label
  nsamples: a threshold controlling number of frames representing a scene in clustering
  """
  print('\n\nsceneCluster start.')
  parent_dir = os.path.dirname(in_file)
  input_list = pd.read_csv(in_file)
  input_names = input_list['path']
  nfiles = len(input_names)
  thres1 = nsamples
  thres2 = 10*thres1
  allfeats = []
  alllabels = []
  for i in range(nfiles):
    fname = input_names[i]
    feats = np.load(os.path.join(parent_dir, fname))
    nframes = feats.shape[0]
    if nframes > thres1 and nframes <= thres2:
      #selid = np.random.choice(nframes, nmax, replace = False)
      selid = np.linspace(0,nframes,thres1, False, dtype=int)
      feats = feats[selid]
    elif nframes > thres2:
      selid = range(0, nframes, 10)
      feats = feats[selid]
    allfeats.append(feats)
    alllabels.append(i*np.ones(feats.shape[0]))
  
  allfeats = np.concatenate(allfeats)
  alllabels = np.concatenate(alllabels)
  # clustering
  allfeats_pca = PCA(2).fit_transform(allfeats)
  _, ids = np.unique(allfeats_pca, axis=0, return_index= True)
  allfeats_un = allfeats_pca[ids]
  alllabels_un = alllabels[ids]
  
  slabel,scount = np.unique(alllabels_un, return_counts=True)
  exclude_scene = slabel[scount < 10]
  selids = np.invert((alllabels_un[:,None]==exclude_scene).sum(axis=1).astype(np.bool))
  allfeats_un = allfeats_un[selids]
  alllabels_un = alllabels_un[selids]
  mycluster = pycluster('meanshift')
  mycluster.fit(allfeats_un, 2, nfiles)
  clabels = mycluster.labels #cluster labels of all samples
  cluster_labels = []
  for i in range(nfiles):
    if i in exclude_scene:
      cluster_labels.append(-1)
    else:
      tscene = clabels[alllabels_un==i]
      cid, bincount = np.unique(tscene, return_counts = True)
      cluster_labels.append(cid[bincount.argmax()])
  input_list['cluster_id'] = cluster_labels
  input_list.to_csv(out_file, index = False)
  nclusters = len(np.unique(cluster_labels))
  print('Num clusters: {}, outliers: {}'.format(nclusters, len(exclude_scene)))
  print('sceneCluster ends.\n\n')
  return nclusters

def CheckFeatLen(data_dir, data_list, max_diff = 3):
  """
  check if augmented data has similar length
  max_diff: maximum difference in length that is acceptable
  """
  lst = pd.read_csv(data_list)
  paths = lst['path'].tolist()
  ids = np.array(lst['clip_id'].tolist())
  min_frames = np.zeros(len(ids), dtype = int)
  id_un = np.unique(ids)
  for id_ in id_un:
    tid = np.where(ids == id_)[0]
    tpaths = [paths[i] for i in tid]
    num_frames = [np.load(os.path.join(data_dir, tpath)).shape[0] for tpath in tpaths]
    num_frames_un = set(num_frames)
    if max(num_frames_un) - min(num_frames_un) > max_diff:
      print('Error! Number of frames mismatch at clip_id %d. One of the transcoding has a problem' % id_)
      print('Feat dir: %s' % data_dir)
      for fr in num_frames_un:
        print('%d frames: %s' % (fr, tpaths[num_frames.index(fr)]))
      raise ValueError('CheckFeatLen error. Video clip id #{}'.format(id_))
    min_frames[tid] = min(num_frames_un)
  lst['min_seq'] = min_frames
  lst.to_csv(data_list, index = False)

def get_cnn_arch(arch_name):
  """return keras cnn architecture
  IN: arch_name   keras supported arch name
  OUT: keras function, prepprocess function, input shape
  """
  supported_archs = ['VGG16', 'InceptionV3', 'InceptionResNetV2', 'ResNet50']
  assert arch_name in supported_archs, 'Error. Unsupported CNN architecture: %s' % arch_name
  if arch_name == 'VGG16':
    base = getattr(apps.vgg16, arch_name)
    prefn = getattr(apps.vgg16, 'preprocess_input')
    last_layer = 'fc2'
    input_shape = (224,224)
  elif arch_name == 'InceptionV3':
    base = getattr(apps.inception_v3, arch_name)
    prefn = getattr(apps.inception_v3, 'preprocess_input')
    last_layer = 'avg_pool'
    input_shape = (299,299)
  elif arch_name == 'ResNet50':
    base = getattr(apps.resnet50, arch_name)
    prefn = getattr(apps.resnet50, 'preprocess_input')
    last_layer = 'avg_pool'
    input_shape = (224,224)
  else:#InceptionResNetV2
    base = getattr(apps.inception_resnet_v2, 'InceptionResNetV2')
    prefn = getattr(apps.inception_resnet_v2, 'preprocess_input')
    last_layer = 'avg_pool'
    input_shape = (299,299)
  base_model = base(weights = 'imagenet', include_top = True)
  model = Model(inputs = base_model.input, outputs = base_model.get_layer(last_layer).output)
  return model, prefn, input_shape

class FeatExtractor(object):
  def __init__(self, batchsize = 15, min_frames = 1, arch = 'ResNet50', verbose = True,
               stride_in_sec = None, **kwargs):
    self.batchsize = batchsize
    self.min_frames = min_frames #min number of frames of a clip to extract feat
    self.verbose = verbose
    self.model, self.prefn, in_shape = get_cnn_arch(arch)
    self.out_dim = self.model.output_shape[1]
    self.video = VideoData(output_size = in_shape, stride_in_sec = stride_in_sec, **kwargs)
  def __del__(self):
    K.clear_session()
    del self.video, self.model
    if self.verbose:
      print('FeatExtractor destructor called.')
    
  def __extract(self, in_file, out_file):
    if self.verbose: print('Process {}'.format(in_file))
    out_dir = os.path.dirname(out_file)
    makeNewDir(out_dir, remove_existing = False)
    try:
      self.video.setup(in_file)
    except Exception as e:
      print('Error! Reading video {} failed: {}'.format(in_file, e))
      return 1
    N = self.video.get_frame_number()
    if N > self.min_frames:
      feats = np.zeros((N, self.out_dim), dtype=np.float32)
      for i in range(0, N, self.batchsize):
        start_  = i
        end_ = min(i+self.batchsize,N)
        x = self.video.get(start_, end_).astype(np.float32)
        x = self.prefn(x)
        feats[start_:end_,...] = self.model.predict(x).squeeze()
      if self.verbose: print('Saving to {}'.format(out_file))
      np.save(out_file, feats)
      return 0
    else:
      if self.verbose: print('Warning: this clip has only {} frames. Ignore it.'.format(N))
      return 1

  def extract(self, input_, output_, logging = True):
    """public method for extracting features
    input_ can be a single video or a csv list of files
    output_ can be a npy path or a dir
    if input_ is a csv list, a csv log will be created under output_
    """
    if self.verbose: print('\n\nfeatExtract start.')
    
    #config io
    if input_.endswith('.txt'):
      multi = True
      input_tab = pd.read_csv(input_)
      input_names = input_tab['path']
      input_tab.rename(columns={'path':'clip'}, inplace=True)
      parent_dir = os.path.dirname(input_)
      input_list = [os.path.join(parent_dir, fname) for fname in input_names]
      makeNewDir(output_, remove_existing = False) #assume output is a dir
      output_names = [x.replace('.','_')+'.npy' for x in input_names]
      output_list = [os.path.join(output_, x) for x in output_names]
      if logging: paths = []
    elif input_.split('.')[-1] in supported_formats:
      multi = False
      input_list = [input_,]
      if output_.endswith('.npy'):
        output_list = [output_,]
      else:
        makeNewDir(output_, remove_existing = False) #assume output is a dir
        output_list = [os.path.join(output_, os.path.basename(input_).replace('.','_')),]
    drop_list = []
    for i, (in_file, out_file) in enumerate(zip(input_list, output_list)):
      if os.path.exists(out_file):
        if self.verbose: print('File exists: %s. Skipping.' % out_file)
        err = 0
      else:
        err = self.__extract(in_file, out_file)
      if err: drop_list.append(i)
      if multi and logging and not err: 
        paths.append(output_names[i])
    if multi and logging:
      if len(drop_list) > 0:
        print('Warning! resnet extract; {} has corrupted ids# {}'.format(input_, drop_list))
        input_tab = input_tab.drop(drop_list)
      input_tab['path'] = paths
      input_tab.to_csv(os.path.join(output_, 'list.txt'), index = False)
    if self.verbose:  print('featExtract ends.\n\n')

class audioMFCC(object):
  """
  class to extract mfcc feature from audio signals
  """
  def __init__(self, winlen = 0.025, winstep = 0.01, ftype = 'both', nworkers = 8, verbose = True):
    self.winlen = winlen
    self.winstep = winstep
    assert ftype in ['mfcc', 'logfbank', 'both'], 'Error. ftype must be "mfcc", "logfbank" or "both"'
    self.ftype = ftype
    self.nworkers = nworkers
    self.verbose = verbose
    
  def extract(self, input_, output_, logging = True):
    """
    public method for extracting mfcc features for audio
    input_ can be a single video or a csv list of files
    output_ can be a npy path or a dir
    if input_ is a csv list, a csv log will be created under output_
    """
    if self.verbose: print('\n\naudioMFCC start.')
    
    #config io
    if input_.endswith('.txt'):
      multi = True
      input_tab = pd.read_csv(input_)
      input_names = input_tab['path']
      input_tab.rename(columns={'path':'clip'}, inplace=True)
      parent_dir = os.path.dirname(input_)
      input_list = [os.path.join(parent_dir, fname) for fname in input_names]
      makeNewDir(output_, remove_existing = False) #assume output is a dir
      output_names = [x.replace('.','_')+'.npy' for x in input_names]
      output_list = [os.path.join(output_, x) for x in output_names]
      if logging: input_tab['path'] = output_names
    elif input_.split('.')[-1] in supported_formats:
      multi = False
      input_list = [input_,]
      if output_.endswith('.npy'):
        output_list = [output_,]
      else:
        makeNewDir(output_, remove_existing = False) #assume output is a dir
        output_list = [os.path.join(output_, os.path.basename(input_).replace('.','_')),]
    # with futures.ThreadPoolExecutor(self.nworkers) as executor:
    #   results = executor.map(self._extract, zip(input_list, output_list))
    # if multi and logging:
    #   for i, res in enumerate(results):
    #     if res == 0:
    #       paths.append(output_names[i])
    #   input_tab['path'] = paths
    #   input_tab.to_csv(os.path.join(output_, 'list.txt'), index = False)
    err_count = 0
    success_ids = []
    with futures.ProcessPoolExecutor(self.nworkers) as executor:
      fut = {executor.submit(extract_audio_feat2file, input_list[i], output_list[i],
                             self.winlen, self.winstep, self.ftype): i for i in
                              range(len(input_list))}
      for future in futures.as_completed(fut):
        res = future.result()
        if res == 0: success_ids.append(fut[future])
        else: err_count += 1
    success_ids.sort()
    if err_count and self.verbose: print('Totally %d audio files failed to mfcc' % err_count)
    if multi and logging:
      output_tab = input_tab.iloc[success_ids]
      output_tab.to_csv(os.path.join(output_, 'list.txt'), index=False)

    if self.verbose:  print('audioMFCC ends.\n\n')
    
  def _extract(self, args):
    audio_file, out_file = args
    if self.verbose: print('Process {}'.format(audio_file))
    try:
      ext = audio_file.split('.')[1]
      if self.verbose:
        print('Process {}'.format(audio_file))
      au = AudioSegment.from_file(audio_file, ext)
      fr = au.frame_rate
      data = np.frombuffer(au._data, np.int16)
      if self.ftype == 'mfcc':
        out = mfcc(data,fr, winlen=self.winlen,winstep=self.winstep,nfft = int(fr*self.winlen))
      elif self.ftype == 'logfbank':
        out = logfbank(data,fr, winlen=self.winlen,winstep=self.winstep,nfft = int(fr*self.winlen))
      elif self.ftype == 'both':
        mfcc_feat = mfcc(data,fr, winlen=self.winlen,winstep=self.winstep,nfft = int(fr*self.winlen))
        fbank_feat = logfbank(data,fr, winlen=self.winlen,winstep=self.winstep,nfft = int(fr*self.winlen))
        out = np.c_[mfcc_feat, fbank_feat]
      #out = np.ravel(fbank_feat) #np.ravel(mfcc_feat)
      np.save(out_file, out)
    except Exception as e:
      print('Error processing {} - {}. Ignore it.'.format(audio_file, e))
      return 1
    return 0