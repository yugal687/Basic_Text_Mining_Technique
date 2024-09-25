
import pandas as pd
from gdelt import gdelt
# from datetime import datetime, timedelta
from datetime import datetime #import of date and time

# newspaper Article
from newspaper import Article

# for nltk
import nltk
from newspaper import Article
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# import os
# # os.kill(os.getpid(), 9)

#  spacy
import spacy
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# # sklearn
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# # matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# newspaper article
article_texts = []
article_ids = []
events = [
{'id': 1, 'url': 'https://www.brecorder.com/news/102936'},
{'id': 2, 'url': 'https://www.kiss925.com/2016/11/14/brendan-dassey-will-released-time-wrestlemania/'}
]
for event in events:
    try:
        url = event['url']
        article_id = event['id']
        article = Article(url)
        article.download()
        article.parse()
        article_texts.append(article.text)
        article_ids.append(article_id)
        print(article_texts)
    except Exception as e:
        print(f"Error processing article {article_id}: {e}")


# # NLTK

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
article_texts = []
article_ids = []
events = [
{'id': 1, 'url': 'https://www.brecorder.com/news/102936'},
{'id': 2, 'url': 'https://www.kiss925.com/2016/11/14/brendan-dassey-will-released-time-wrestlemania/'}
]

def preprocess_text(text):
    text = ''.join([char.lower() if char.isalpha() or char.isspace() else ' ' for char in text])
    words = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

for event in events:
    try:
        url = event['url']
        article_id = event['id']
        article = Article(url)
        article.download()
        article.parse()
        processed_text = preprocess_text(article.text)
        article_texts.append(processed_text)
        article_ids.append(article_id)
        print(article_texts)
    except Exception as e:
        print(f"Error processing article {article_id}: {e}")


# # spacy
nlp = spacy.load('en_core_web_sm')
article_texts = [
'tokyo japan nikkei share average edged thursday morning strong china data helped firm high exposure world second largest economy japanese supplier apple inc mostly recovered drop disappointing iphone sale figure benchmark kicked positive territory data showed growth china factory sector hit two year high january helping komatsu ltd advance percent nikkei china outperformed gain percent hsbc flash purchasing manager index pmi rose january highest since january point level show accelerating growth sector previous month midday break nikkei edged percent breaking back level hitting three week closing low wednesday third straight day decline china helped basically three day loss created nice dip buying opportunity easy get today said masato futoi head cash equity trading tokai tokyo security underlying tone still bullish even bad news apple whatever hit stock hard coming pressure early trading apple supplier murata manufacturing co ltd foster electric co ltd taiyo yuden co ltd midday break however ibiden co ltd make printed circuit board iphone slid percent apple said shipped million iphones december quarter roughly million predicted wall street analyst sent share skidding percent extended trading investor cut exposure stock iphone outpaced competitor supplier samsung electronics co ltd galaxy note ii world popular smartphone whose sale contributed percent rise korean company operating profit december quarter wacom co ltd maker galaxy touchscreen jumped percent hiking full year operating profit forecast nearly fifth billion yen million citing strong sale back weaker yen quarterly earnings figure beginning trickle japan earnings reporting moving higher gear next week lack incentive moment everyone waiting result said masayuki doshida senior market analyst rakuten security yaskawa electric corp fell much percent one month low company reported percent drop operating profit nine month december hurt weak demand servo motor china europe midday pared loss percent yen toshiba corp also focus gaining percent nikkei business daily said discus forming joint venture general electric co develop sell combined cycle gas turbine','tokyo japan nikkei share average edged thursday morning strong china data helped firm high exposure world second largest economy japanese supplier apple inc mostly recovered drop disappointing iphone sale figure benchmark kicked positive territory data showed growth china factory sector hit two year high january helping komatsu ltd advance percent nikkei china outperformed gain percent hsbc flash purchasing manager index pmi rose january highest since january point level show accelerating growth sector previous month midday break nikkei edged percent breaking back level hitting three week closing low wednesday third straight day decline china helped basically three day loss created nice dip buying opportunity easy get today said masato futoi head cash equity trading tokai tokyo security underlying tone still bullish even bad news apple whatever hit stock hard coming pressure early trading apple supplier murata manufacturing co ltd foster electric co ltd taiyo yuden co ltd midday break however ibiden co ltd make printed circuit board iphone slid percent apple said shipped million iphones december quarter roughly million predicted wall street analyst sent share skidding percent extended trading investor cut exposure stock iphone outpaced competitor supplier samsung electronics co ltd galaxy note ii world popular smartphone whose sale contributed percent rise korean company operating profit december quarter wacom co ltd maker galaxy touchscreen jumped percent hiking full year operating profit forecast nearly fifth billion yen million citing strong sale back weaker yen quarterly earnings figure beginning trickle japan earnings reporting moving higher gear next week lack incentive moment everyone waiting result said masayuki doshida senior market analyst rakuten security yaskawa electric corp fell much percent one month low company reported percent drop operating profit nine month december hurt weak demand servo motor china europe midday pared loss percent yen toshiba corp also focus gaining percent nikkei business daily said discus forming joint venture general electric co develop sell combined cycle gas turbine', 'brendan dassey released time wrestlemania judge today ordered immediate release making murderer brendan dassey guaranteeing home time watch wrestlemania next year fan show remember dassey teen put jail murder theresa halbach year old finally receiving justice netflix hit show brought story mainstream judge order immediate release brendan dassey whose case featured netflix making murderer http co duecc dhot pic twitter com j vpoxtge bbc breaking news bbcbreaking november dailymail report august judge ruled police tricked intellectually disabled teenager describing helped rape stab shoot dismember halbach uncle steven avery order dassey supervised release immediate noon tuesday provide federal probation parole office address planned live attorney steve drizin would say dassey plan live dassey conviction based confession gave investigator working prosecution http co ckbfewxlaz abbey perano abbeyperano november dassey providing address intended residence tuesday afternoon barred contacting uncle steven avery']
def spacy_preprocess(texts):
    processed_texts = []
    for doc in nlp.pipe(texts):
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        processed_texts.append(' '.join(tokens))
    return processed_texts
processed_texts = spacy_preprocess(article_texts)
count_vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')
tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
count_matrix = count_vectorizer.fit_transform(processed_texts)
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(count_matrix)
nmf = NMF(n_components=5, random_state=42)
nmf.fit(tfidf_matrix)
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
print("LDA Topics:")
display_topics(lda, count_vectorizer.get_feature_names_out(), 10)
print("\nNMF Topics:")
display_topics(nmf, tfidf_vectorizer.get_feature_names_out(), 10)
lda_topic_distribution = lda.transform(count_matrix)
nmf_topic_distribution = nmf.transform(tfidf_matrix)
print("\nLDA Topic Distribution for the first document:")
print(lda_topic_distribution[0])
print("\nNMF Topic Distribution for the first document:")
print(nmf_topic_distribution[0])


# # Nltk
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
article_texts = [
'tokyo japan nikkei share average edged thursday morning strong china data helped firm high exposure world second largest economy japanese supplier apple inc mostly recovered drop disappointing iphone sale figure benchmark kicked positive territory data showed growth china factory sector hit two year high january helping komatsu ltd advance percent nikkei china outperformed gain percent hsbc flash purchasing manager index pmi rose january highest since january point level show accelerating growth sector previous month midday break nikkei edged percent breaking back level hitting three week closing low wednesday third straight day decline china helped basically three day loss created nice dip buying opportunity easy get today said masato futoi head cash equity trading tokai tokyo security underlying tone still bullish even bad news apple whatever hit stock hard coming pressure early trading apple supplier murata manufacturing co ltd foster electric co ltd taiyo yuden co ltd midday break however ibiden co ltd make printed circuit board iphone slid percent apple said shipped million iphones december quarter roughly million predicted wall street analyst sent share skidding percent extended trading investor cut exposure stock iphone outpaced competitor supplier samsung electronics co ltd galaxy note ii world popular smartphone whose sale contributed percent rise korean company operating profit december quarter wacom co ltd maker galaxy touchscreen jumped percent hiking full year operating profit forecast nearly fifth billion yen million citing strong sale back weaker yen quarterly earnings figure beginning trickle japan earnings reporting moving higher gear next week lack incentive moment everyone waiting result said masayuki doshida senior market analyst rakuten security yaskawa electric corp fell much percent one month low company reported percent drop operating profit nine month december hurt weak demand servo motor china europe midday pared loss percent yen toshiba corp also focus gaining percent nikkei business daily said discus forming joint venture general electric co develop sell combined cycle gas turbine','tokyo japan nikkei share average edged thursday morning strong china data helped firm high exposure world second largest economy japanese supplier apple inc mostly recovered drop disappointing iphone sale figure benchmark kicked positive territory data showed growth china factory sector hit two year high january helping komatsu ltd advance percent nikkei china outperformed gain percent hsbc flash purchasing manager index pmi rose january highest since january point level show accelerating growth sector previous month midday break nikkei edged percent breaking back level hitting three week closing low wednesday third straight day decline china helped basically three day loss created nice dip buying opportunity easy get today said masato futoi head cash equity trading tokai tokyo security underlying tone still bullish even bad news apple whatever hit stock hard coming pressure early trading apple supplier murata manufacturing co ltd foster electric co ltd taiyo yuden co ltd midday break however ibiden co ltd make printed circuit board iphone slid percent apple said shipped million iphones december quarter roughly million predicted wall street analyst sent share skidding percent extended trading investor cut exposure stock iphone outpaced competitor supplier samsung electronics co ltd galaxy note ii world popular smartphone whose sale contributed percent rise korean company operating profit december quarter wacom co ltd maker galaxy touchscreen jumped percent hiking full year operating profit forecast nearly fifth billion yen million citing strong sale back weaker yen quarterly earnings figure beginning trickle japan earnings reporting moving higher gear next week lack incentive moment everyone waiting result said masayuki doshida senior market analyst rakuten security yaskawa electric corp fell much percent one month low company reported percent drop operating profit nine month december hurt weak demand servo motor china europe midday pared loss percent yen toshiba corp also focus gaining percent nikkei business daily said discus forming joint venture general electric co develop sell combined cycle gas turbine', 'brendan dassey released time wrestlemania judge today ordered immediate release making murderer brendan dassey guaranteeing home time watch wrestlemania next year fan show remember dassey teen put jail murder theresa halbach year old finally receiving justice netflix hit show brought story mainstream judge order immediate release brendan dassey whose case featured netflix making murderer http co duecc dhot pic twitter com j vpoxtge bbc breaking news bbcbreaking november dailymail report august judge ruled police tricked intellectually disabled teenager describing helped rape stab shoot dismember halbach uncle steven avery order dassey supervised release immediate noon tuesday provide federal probation parole office address planned live attorney steve drizin would say dassey plan live dassey conviction based confession gave investigator working prosecution http co ckbfewxlaz abbey perano abbeyperano november dassey providing address intended residence tuesday afternoon barred contacting uncle steven avery'
]
def spacy_preprocess(texts):
    stop_words = set(stopwords.words('english'))
    processed_texts = []
    for doc in nlp.pipe(texts):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in stop_words]
        processed_texts.append(tokens)
    return processed_texts
processed_texts = spacy_preprocess(article_texts)
flattened_texts = [' '.join(tokens) for tokens in processed_texts]
analyzer = SentimentIntensityAnalyzer()
def analyze_sentiment(texts):
    sentiments = []
    for text in texts:
        sentiment = analyzer.polarity_scores(text)
        sentiments.append(sentiment)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
    return sentiments
sentiments = analyze_sentiment(flattened_texts)
def extract_ngrams(tokens_list, n=2):
    ngrams=[]
    for tokens in tokens_list:
        ngrams += list(nltk.ngrams(tokens, n))
    return ngrams
bigrams = extract_ngrams(processed_texts, n=2)
trigrams = extract_ngrams(processed_texts, n=3)
bigram_counter = Counter(bigrams)
trigram_counter = Counter(trigrams)
print("\nMost Common Bigrams:")
for bigram, count in bigram_counter.most_common(5):
    print(f"{bigram}: {count}")
print("\nMost Common Trigrams:")
for trigram, count in trigram_counter.most_common(5):
    print(f"{trigram}: {count}")


# # matplotlib
nlp = spacy.load('en_core_web_sm')
article_texts = [
'tokyo japan nikkei share average edged thursday morning strong china data helped firm high exposure world second largest economy japanese supplier apple inc mostly recovered drop disappointing iphone sale figure benchmark kicked positive territory data showed growth china factory sector hit two year high january helping komatsu ltd advance percent nikkei china outperformed gain percent hsbc flash purchasing manager index pmi rose january highest since january point level show accelerating growth sector previous month midday break nikkei edged percent breaking back level hitting three week closing low wednesday third straight day decline china helped basically three day loss created nice dip buying opportunity easy get today said masato futoi head cash equity trading tokai tokyo security underlying tone still bullish even bad news apple whatever hit stock hard coming pressure early trading apple supplier murata manufacturing co ltd foster electric co ltd taiyo yuden co ltd midday break however ibiden co ltd make printed circuit board iphone slid percent apple said shipped million iphones december quarter roughly million predicted wall street analyst sent share skidding percent extended trading investor cut exposure stock iphone outpaced competitor supplier samsung electronics co ltd galaxy note ii world popular smartphone whose sale contributed percent rise korean company operating profit december quarter wacom co ltd maker galaxy touchscreen jumped percent hiking full year operating profit forecast nearly fifth billion yen million citing strong sale back weaker yen quarterly earnings figure beginning trickle japan earnings reporting moving higher gear next week lack incentive moment everyone waiting result said masayuki doshida senior market analyst rakuten security yaskawa electric corp fell much percent one month low company reported percent drop operating profit nine month december hurt weak demand servo motor china europe midday pared loss percent yen toshiba corp also focus gaining percent nikkei business daily said discus forming joint venture general electric co develop sell combined cycle gas turbine','tokyo japan nikkei share average edged thursday morning strong china data helped firm high exposure world second largest economy japanese supplier apple inc mostly recovered drop disappointing iphone sale figure benchmark kicked positive territory data showed growth china factory sector hit two year high january helping komatsu ltd advance percent nikkei china outperformed gain percent hsbc flash purchasing manager index pmi rose january highest since january point level show accelerating growth sector previous month midday break nikkei edged percent breaking back level hitting three week closing low wednesday third straight day decline china helped basically three day loss created nice dip buying opportunity easy get today said masato futoi head cash equity trading tokai tokyo security underlying tone still bullish even bad news apple whatever hit stock hard coming pressure early trading apple supplier murata manufacturing co ltd foster electric co ltd taiyo yuden co ltd midday break however ibiden co ltd make printed circuit board iphone slid percent apple said shipped million iphones december quarter roughly million predicted wall street analyst sent share skidding percent extended trading investor cut exposure stock iphone outpaced competitor supplier samsung electronics co ltd galaxy note ii world popular smartphone whose sale contributed percent rise korean company operating profit december quarter wacom co ltd maker galaxy touchscreen jumped percent hiking full year operating profit forecast nearly fifth billion yen million citing strong sale back weaker yen quarterly earnings figure beginning trickle japan earnings reporting moving higher gear next week lack incentive moment everyone waiting result said masayuki doshida senior market analyst rakuten security yaskawa electric corp fell much percent one month low company reported percent drop operating profit nine month december hurt weak demand servo motor china europe midday pared loss percent yen toshiba corp also focus gaining percent nikkei business daily said discus forming joint venture general electric co develop sell combined cycle gas turbine', 'brendan dassey released time wrestlemania judge today ordered immediate release making murderer brendan dassey guaranteeing home time watch wrestlemania next year fan show remember dassey teen put jail murder theresa halbach year old finally receiving justice netflix hit show brought story mainstream judge order immediate release brendan dassey whose case featured netflix making murderer http co duecc dhot pic twitter com j vpoxtge bbc breaking news bbcbreaking november dailymail report august judge ruled police tricked intellectually disabled teenager describing helped rape stab shoot dismember halbach uncle steven avery order dassey supervised release immediate noon tuesday provide federal probation parole office address planned live attorney steve drizin would say dassey plan live dassey conviction based confession gave investigator working prosecution http co ckbfewxlaz abbey perano abbeyperano november dassey providing address intended residence tuesday afternoon barred contacting uncle steven avery'
]
def spacy_preprocess(texts):
    stop_words = set(stopwords.words('english'))
    processed_texts = []
    for doc in nlp.pipe(texts):
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in stop_words]
        processed_texts.append(' '.join(tokens))
    return processed_texts
processed_texts = spacy_preprocess(article_texts)
combined_text = ' '.join(processed_texts)
wordcloud = WordCloud(width=900, height=600, background_color='black', colormap='viridis', max_words=120).generate(combined_text)
plt.figure(figsize=(20, 9))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()