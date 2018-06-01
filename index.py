from __future__ import print_function
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time

import chainer
from chainer import cuda, Variable, serializers
from net import *

import sys
import os
import json
from flask import Flask, request, render_template, Response
from werkzeug.routing import BaseConverter
import settings

app = Flask(__name__)


class RegexConverter(BaseConverter):
  def __init__(self, url_map, *items):
    super( RegexConverter, self).__init__(url_map)
    self.regex = items[0]

app.url_map.converters['regex'] = RegexConverter


# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')

@app.route('/')
def index():
  name = "TEST"
  return render_template('index.html', title='chainer-gogh test', name=name)

@app.route('/models')
def models():
  files = os.listdir( 'models' );
  list = []
  for file in files:
    if file.endswith( '.model' ):
      list.append( file )
  arr = json.dumps( list )

  return Response( response=arr, content_type='application/json' )

@app.route('/<regex("[0-9]*"):uid>.jpg')
def image(uid):
  #return 'tmp/' + uid + '.jpg'
  IMAGEFILE = 'tmp/output_%s.jpg' % (uid)
  f = open( IMAGEFILE, 'rb' )
  image = f.read()
  if settings.leaveimage == False:
    os.remove( IMAGEFILE )
  return Response( response=image, content_type='image/jpeg' )

@app.route('/post', methods = ['POST'])
def post():
  start = time.time()

  INPUTFILE = 'tmp/input_%d.jpg' % (start)
  OUTPUTFILE = 'tmp/output_%d.jpg' % (start)

  filebuf = request.files.get( 'image' )
  stream = filebuf.stream

  stylemodel = 'models/seurat.model'
  if request.form['stylemodel'] :
    stylemodel = 'models/%s.model' % request.form['stylemodel']

  f = open( INPUTFILE, 'wb' )
  f.write( stream.read() )

  # chainer
  model = FastStyleNet()
  serializers.load_npz(stylemodel, model)
  if settings.gpu >= 0:
    cuda.get_device(settings.gpu).use()
    model.to_gpu()
  xp = np if settings.gpu < 0 else cuda.cupy

  original = Image.open( INPUTFILE ).convert( 'RGB' )
  image = np.asarray( original, dtype=np.float32 ).transpose( 2, 0, 1 )
  image = image.reshape((1,) + image.shape)
  if settings.padding > 0:
    image = np.pad( image, [[0,0],[0,0],[settings.padding,settings.padding],[settings.padding,settings.padding]], 'symmetric')
  image = xp.asarray(image)

  x = Variable(image)
  y = model(x)
  result = cuda.to_cpu(y.data)
  if settings.padding > 0:
    result = result[:, :, settings.padding:-settings.padding, settings.padding:-settings.padding]
  result = np.uint8(result[0].transpose((1,2,0)))
  med = Image.fromarray(result)
  if settings.median_filter > 0:
    med = med.filter(ImageFilter.MedianFilter(settings.median_filter))
  if settins.keep_colors:
    med = original_colors(original, med)
  end = time.time()
  print(end - start, 'sec')
  med.save( OUTPUTFILE )

  if settings.leaveimage == False:
    os.remove( INPUTFILE )

  return '%d' % start


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=settings.port)
