from setuptools import setup

setup(name='preprocessing_tele',
      version='1.0',
      author='Francesco Dalmonte',
      packages=['preprocessing_tele',
                'preprocessing_tele.test'],
      scripts=['bin/create_cropsDataset.py',
               'bin/create_cropsDs_Multimodal.py',
               'bin/run_inference_v2.py',
               'bin/run_inference.py',
               'bin/visualize_labelling.py'],
      license='LICENSE.txt'
      )