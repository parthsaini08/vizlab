from distutils.core import setup
setup(
  name = 'vizlab',        
  packages = ['vizlab'],
  version = '0.1',      
  license='MIT',        
  description = 'Library for visualizing machine learning models and for Data Preprocessing',  
  author = 'Parth Saini',                   
  author_email = 'parth08.saini@gmail.com',      
  url = 'https://github.com/parthsaini08/vizlab',  
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    
  keywords = ['VISUALIZE', 'MODELS'],  
  install_requires=[           
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)