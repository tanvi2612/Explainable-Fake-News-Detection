# Steps to run extention
- Go to the detectr directory
- Download the model from the given Gdrive link "https://drive.google.com/file/d/1Yn3l_quOMV-ncLYzVU06WZ3372VvE10Y/view?usp=sharing" (at the same directory as the app.py file)
- Install the required packages
## Creating Extension For Chrome

- Click on manage extension in chrome settings
- Click on load unpacked
- Select the folder "Detectr" contatining "manifest.json" 
- Run app.py


 


# Collecting Dataset
## FakeNewsNet
Clone github repo "https://github.com/KaiDMML/FakeNewsNet.git"

## Overview  


The minimalistic version of latest dataset provided in this repo (located in `dataset` folder) include following files:

 - `politifact_fake.csv` -  Samples related to fake news collected from PolitiFact 
 - `politifact_real.csv` -  Samples related to real news collected  from PolitiFact 
 - `gossipcop_fake.csv` - Samples related to fake news collected from GossipCop
  - `gossipcop_real.csv` - Samples related to real news collected from GossipCop

Each of the above CSV files is comma separated file and have the following columns

 - `id` - Unique identifider for each news
 - `url` - Url of the article from web that published that news 
 - `title` - Title of the news article
 - `tweet_ids` - Tweet ids of tweets sharing the news. This field is list of tweet ids separated by tab.

## Installation    

###  Requirements:
 Data download scripts are writtern in python and requires `python 3.6 +` to run.
 
Twitter API keys are used for collecting data from Twitter.  Make use of the following link to get Twitter API keys    
https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html   

Script make use of keys from  _tweet_keys_file.json_ file located in `code/resources` folder. So the API keys needs to be updated in `tweet_keys_file.json` file.  Provide the keys as array of JSON object with attributes `app_key,app_secret,oauth_token,oauth_token_secret` as mentioned in sample file.

Install all the libraries in `requirements.txt` using the following command
    
    pip install -r requirements.txt


###  Configuration:

 FakeNewsNet contains 2 datasets collected using ground truths from _Politifact_ and _Gossipcop_.  
    
The `config.json` can be used to configure and collect only certain parts of the dataset. Following attributes can be configured    
  
 - **num_process** - (default: 4) This attribute indicates the number of parallel processes used to collect data.    
 - **tweet_keys_file** - Provide the number of keys available configured in tweet_keys_file.txt file       
 - **data_collection_choice** - It is an array of choices of various parts of the dataset. Configure accordingly to download only certain parts of the dataset.       
   Available values are  
     {"news_source": "politifact", "label": "fake"},{"news_source": "politifact", "label":    "real"}, {"news_source": "gossipcop", "label": "fake"},{"news_source": "gossipcop", "label": "real"}  
  
 - **data_features_to_collect** - FakeNewsNet has multiple dimensions of data (News + Social). This configuration allows one to download desired dimension of the dataset. This is an array field and can take following values.  
	              
	 - **news_articles** : This option downloads the news articles for the dataset.  
     - **tweets** : This option downloads tweets objects posted sharing the news in Twitter. This makes use of Twitter API to download tweets.  
     - **retweets**: This option allows to download the retweets of the tweets provided in the dataset.  



## Running Code

Inorder to collect data set fast, code makes user of process parallelism and to synchronize twitter key limitations across mutiple python processes, a lightweight flask application is used as keys management server.
Execute the following commands inside `code` folder,

    nohup python -m resource_server.app &> keys_server.out&

The above command will start the flask server in port 5000 by default.

**Configurations should be done before proceeding to the next step !!**

Execute the following command to start data collection,

    nohup python main.py &> data_collection.out&

Logs are wittern in the same folder in a file named as `data_collection_<timestamp>.log` and can be used for debugging purposes.

The dataset will be downloaded in the directory provided in the `config.json` and progress can be monitored in `data_collection.out` file. 

### Dataset Structure
The downloaded dataset will have the following  folder structure,
```bash
├── gossipcop
│   ├── fake
│   │   ├── gossipcop-1
│   │	│	├── news content.json
│   │	│	├── tweets
│   │	│	│	├── 886941526458347521.json
│   │	│	│	├── 887096424105627648.json
│   │	│	│	└── ....		
│   │	│  	└── retweets
│   │	│		├── 887096424105627648.json
│   │	│		├── 887096424105627648.json
│   │	│		└── ....
│   │	└── ....			
│   └── real
│      ├── gossipcop-1
│      │	├── news content.json
│      │	├── tweets
│      │	└── retweets
│		└── ....		
├── politifact
│   ├── fake
│   │   ├── politifact-1
│   │   │	├── news content.json
│   │   │	├── tweets
│   │   │	└── retweets
│   │	└── ....		
│   │
│   └── real
│      ├── poliifact-2
│      │	├── news content.json
│      │	├── tweets
│      │	└── retweets
│      └── ....					

```
**News Content**

`news content.json`:
This json includes all the meta information of the news articles collected using the provided news source URLs. This is a JSON object with attributes including:

 - `text` is the text of the body of the news article. 
 - `images` is a list of the URLs of all the images in the news article web page. 
 - `publish date`  indicate the date that news article is published.


# Pretraining on Common Crawl
- extract common crawl data:  aws s3 cp s3://commoncrawl/crawl-data/CC-NEWS/2021/03/CC-NEWS-20210305051940-00557.warc.gz . --no-sign-request
- To extract all the zip files python3 -m warcat extract ../cc-eng/CC-NEWS-20210305082016-00559.warc.gz
- To store only the English language data  python3 save_files.py > listfiles4.txt 
- Finally to extract data and store in articles.txt python3 extract_text.py
- python3 run_mlm.py --model_name_or_path bert-base-cased --train_file ./articles.txt --line_by_line --do_train --output_dir ./out --per_device_train_batch_size 1
- make sure that articles and run_mlm.py are in same folder 
- run_mlm.py can be downloaded from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/

# Finetuning pretrained model
- Take the pretrained model(obtained from the previous step) and the dataset extracted previously
- Install required dependencies
- Create a csv file through the directory structure having label as 0 (fake) and 1 (real).
- Run train1.py in the src directory(set location of pretrained model and data as you wish)
- The code will return a few models. To find the best run score1.py to get the best cutoffs and model.
- Run the extension again using the above steps for better results.
