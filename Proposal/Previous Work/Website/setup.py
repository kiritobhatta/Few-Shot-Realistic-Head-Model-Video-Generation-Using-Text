from setuptools import setup

setup(name='gan',
      version='0.2',
      description='Produces speech-driven faces',
      packages=['gan'],
      package_dir={'gan': 'gan'},
      package_data={'gan': ['data/*.dat']},
      install_requires=[
          'numpy',
          'scipy',
          'scikit-video',
          'scikit-image',
          'ffmpeg-python',
          'torch',
          'face-alignment',
          'torchvision',
          'pydub',
          'Flask',
          'gTTS',
          'python-vlc',
          'js2py'
          ],
      zip_safe=False)
