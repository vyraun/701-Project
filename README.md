# 701-Project
Quora Dataset, Semantic Textual Similarity

* Tokenize: python quora_tokenize.py quora_duplicate_questions.tsv
* wget http://nlp.stanford.edu/data/glove.6B.zip 
* unzip glove.6B.zip 
* unzip all_splits.zip
* https://github.com/facebookresearch/faiss/blob/master/INSTALL.md
* https://stackoverflow.com/questions/36659453/intel-mkl-fatal-error-cannot-load-libmkl-avx2-so-or-libmkl-def-so
* https://docs.continuum.io/mkl-optimizations/


* Seting up on aws: conda install faiss-cpu -c pytorch
  >>> import nltk
  >>> nltk.download('stopwords')
  >>> nltk.download('punkt')
