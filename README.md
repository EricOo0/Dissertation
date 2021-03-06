# Dissertation
My master Dissertation in NTU
* It is an NLP related dissertation
* The topic is Online News Analytics based on AI Techniques
* The main content is to collect a natural disaster dataset and perform news classification tasks on different feature representation methods and classifier

## Usage

* ๐ Dataset๏ผ

  โ	BBC_news is an dataset contains news from bbc website.

  โ	disaster_news_dataset.csv is the collected dataset about natrual disaster.

  โ	path:  code/dataset/disaster_news_dataset.csv

  โ	content:

  โ    <img src="image/image-20211024131756544.png" alt="image-20211024131756544" style="zoom:30%;" />

  โ	Distribution:

  <img src="image/image-20211024131937924.png" alt="image-20211024131937924" style="zoom:30%;" />

  <img src="image/image-20211024131952624.png" alt="image-20211024131952624" style="zoom:30%;" />

  <img src="image/image-20211024132019640.png" alt="image-20211024132019640" style="zoom:30%;" />

  โ		data_extraction.py:  is used to separate the dataset into test_set,trainning_set and validation_set. you might have to change the loading path before you use.

  โ		build_w2v.py: is to transfer the word2vec to the type that could be used in model.  **<u>You should download GoogleNews-vectors-negative300.bin before using this py</u>**

* ๐ธ๏ธBow+LR

  โ	path: code/bow+LR

  โ	train and test command : **python3 main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

* ๐ธ๏ธBow+SVM

  โ	path: code/bow+SVM

  โ	train and test command : **python3 main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

* ๐ธ๏ธBow+Feed-forward

  โ	path: code/bow+feedforward

  โ	train and test command : **python3 main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

<img src="image/image-20211211003809154.png" alt="image-20211211003809154" style="zoom:50%;" />

* ๐ธ๏ธTF-IDF+LR

  โ	path: code/bow+LR

  โ	modify the code to use TF-IDF

  โ	train and test command : **python3 main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

* ๐ธ๏ธTF-IDF+SVM

  โ	path: code/bow+SVM

  โ	modify the code to use TF-IDF

  โ	train and test command : **python3 main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

* ๐ธ๏ธTF-IDF+Feed-forward

  โ	path: code/bow+feedforward

  โ	modify the code to use TF-IDF

  โ	train and test command : **python3 main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

<img src="image/image-20211211003846660.png" alt="image-20211211003846660" style="zoom:50%;" />

* ๐ธ๏ธLDA+LR

  โ	path: code/LDA+LR

  โ	train and test command : **python3 main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

* ๐ธ๏ธLDA+SVM

  โ	path: code/LDA+SVM

  โ	train and test command : **python3 main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

* ๐ธ๏ธLDA+Feed-forward

  โ	path: code/LDA+feedforward

  โ	train and test command : **python3 main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

<img src="image/image-20211211003943614.png" alt="image-20211211003943614" style="zoom:50%;" />

<img src="image/image-20211211004128139.png" alt="image-20211211004128139" style="zoom:50%;" />

* ๐ธ๏ธ Text-CNN๏ผ

  โ	path๏ผcode/text_cnn

  โ	train and test command : **python main.py** 

  โ	test criteria is: accuracy, precise,recall,f1-score.

  โ	you can modify the code to use non-pretrain embedding / word2vec embedding /glove embedding	

  โ	**you should download glove.840B.300d.txt and put it in correct path before you use it.**

* ๐ธ๏ธ LSTM๏ผ

  โ	path: code/lstm

  โ	train and test command : **python main.py** 	

  โ	test criteria is: accuracy, precise,recall,f1-score.

  โ	you can also modify the code to use non-pretrain embedding / word2vec embedding /glove embedding	    

  โ	when using non-pretrain embedding, you might have to modify the setting.py and model.py

  โ	**you should download glove.840B.300d.txt and put it in correct path before you use it.**

* ๐ธ๏ธ Transformer๏ผ

  โ	based on huggingface transformer

  โ	path: code/transformer

  โ	train and test command: 

  ```
  export TASK_NAME=test_classifify
  python run_glue.py \
    --model_name_or_path bert-base-cased \
    --train_file ../dataset/training_set.csv \
    --validation_file ../dataset/validation_set.csv \
    --test_file ../dataset/test_set.csv \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 256 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 32 \
    --overwrite_output_dir True \
    --output_dir ./$TASK_NAME/
  ```

  use Bert-base-case model to perform classification tasks;

* ๐Other:

  โ	these models will be train 10 times to evaluate the mean performance by default, you can modify the code to change.

  

* ๐ Result๏ผ

  <img src="image/image-20211024135421500.png" alt="image-20211024135421500" style="zoom:50%;" />

  <img src="image/image-20211211004041421.png" alt="image-20211211004041421" style="zoom:50%;" />

  <img src="image/image-20211024135439948.png" alt="image-20211024135439948" style="zoom:50%;" />

  <img src="image/image-20211211004108314.png" alt="image-20211211004108314" style="zoom:50%;" />