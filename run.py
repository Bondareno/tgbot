import re
import pandas as pd
import nltk
from nltk import punkt
from nltk import *
from bs4 import BeautifulSoup
from pymystem3 import Mystem
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import requests
import telebot
from telebot import *
import io
import traceback

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
sns.color_palette("tab10")
sns.set(style="whitegrid")

TOKEN='6954819921:AAHVZw4_nZ36dU7iqnucWBbsyPeP4CSgrUY'
bot = telebot.TeleBot(TOKEN)
morph = pymorphy2.MorphAnalyzer()
m = Mystem(disambiguation=False)

def extract_data(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        text = soup.get_text()

        date_match = re.search(r'\d+\s+[A-Za-z]+\s+\d+', text)
        date = date_match.group() if date_match else None

        time_match = re.search(r'\d+:\d+', text)
        time = time_match.group() if time_match else None

        speakers = re.findall(r'—\s+([^\n]+)', text)
        dialogues = re.findall(r'—\s+([^\n]+)\n', text)

        text_extracted = speakers + dialogues
        tdf = pd.DataFrame(text_extracted)
        tdf = tdf.rename(columns={0:'text'})
    
    except FileNotFoundError:
        print("Файл пока нет.")
    return tdf
    

def len_sent(tdf):
    tdf['sent_count'] = tdf['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
    tdf['word_count'] = tdf['text'].apply(lambda x:len(nltk.word_tokenize(x)))
    tdf['word_per_sent'] = tdf['word_count']/tdf['sent_count']
    word_per_sent = tdf['word_per_sent'].mean()
    return round(word_per_sent) 


def plot_dirty(tdf, ax):
    count_vect_total = CountVectorizer(ngram_range=(1,1), min_df=5)

    corpus_total = [x for x in tdf['text'].fillna(' ') if len(str(x)) > 0]

    corpus_total_fit = count_vect_total.fit_transform(corpus_total)
    total_counts = pd.DataFrame(corpus_total_fit.toarray(), columns=count_vect_total.get_feature_names_out()).sum()
    ngram_total_df = pd.DataFrame(total_counts, columns=['counts'])

    
    ngram_total_df = ngram_total_df.sort_values(by='counts', ascending=False)
    
    plot_dirty = sns.barplot(x="counts",
                             y=ngram_total_df.head(20).index,
                             ax=ax,
                             data=ngram_total_df.head(20),
                             hue=ngram_total_df.head(20).index,
                             legend=False)
    
    ax.set_title('TОП СЛОВ БЕЗ ЧИСТКИ')
    return plot_dirty

def lemmatize(wrds, m):
    res = []
    for wrd in wrds:
        p = m.parse(wrd)[0]
        res.append(p.normal_form)
        
    return res

def tokenize(text, stoplst):
    without_stop_words = []
    txxxt = nltk.word_tokenize(text)
    for word in txxxt:
        if len(word) == 1:
            continue
        if word.lower() not in stoplst:
            without_stop_words.append(word)
    return without_stop_words



def plot_clean(tdf,stop_words, ax):
    text = str(tdf['text'].values)#.astype('str')
    tokens = word_tokenize(text)
    tokens = lemmatize(tokens, morph)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    count_vect_total = CountVectorizer(ngram_range=(1,2),min_df=5)
    corpus_total = [x for x in filtered_tokens if (str(x) not in stop_words) and (len(x)>3)]


    corpus_total_fit = count_vect_total.fit_transform(corpus_total)#(corpus_total)

    total_counts = pd.DataFrame(corpus_total_fit.toarray(),columns=count_vect_total.get_feature_names_out()).sum()
    ngram_total_df = pd.DataFrame(total_counts,columns=['counts'])

    ngram_total_df = ngram_total_df.sort_values(by=['counts'],ascending=False)
    
    plot_clean = sns.barplot(x="counts",
                            y=ngram_total_df.head(20).index,
                            data=ngram_total_df.head(20),
                            
                            hue=ngram_total_df.head(20).index,
                            legend=False, ax=ax)
    ax.set_title('TОП СЛОВ ПОСЛЕ ЧИСТКИ')
                    
    return plot_clean



def uniq_words_share(tdf,stoplst):
    txt = str(tdf['text'].values)
    tokenized_text = tokenize(txt, stop_words)
    tokens = lemmatize(tokenized_text, morph)
    words = len(tokens)
    uniq_words = len(set(tokens))
    return (uniq_words/words) * 100


def process_and_visualize(stop_words, tdf):
    text = str(tdf['text'].values)
    
    def pre_process(text):
        text = re.sub(r"</?.*?>", " <> ", str(text))
        text = re.sub(r"(\\d|\\W)+", "", text)
        text = re.sub(r'[^а-яА-Я\s]+', ' ', text)
        text = text.lower()
        text = re.sub(r"\r\n", " ", text)
        text = re.sub(r"\xa0", " ", text)
        sub_str = 'sfgff'
        text = text[:text.find(sub_str)]
        return text

    morph = pymorphy2.MorphAnalyzer()

    cleaned_text = pre_process(text)
    tokenized_text = tokenize(cleaned_text, stop_words)
    lemmatized_text = lemmatize(tokenized_text, morph)

    Fdist = FreqDist(lemmatized_text)

    not_most_common = Fdist.most_common()[-21:-1]
    rare_words = [i[0] for i in not_most_common]

    

    return ', '.join(str(w) for w in rare_words)


# top pos:
def plot_pos(tdf, ax):
    pos_counter_bk = Counter()
    bk = str(tdf['text'].values)

    for sentence in sent_tokenize(bk, language="russian"):
        doc = m.analyze(sentence) 
        for word in doc: 
                if "analysis" not in word or len(word["analysis"]) == 0: 
                    continue

                gr = word["analysis"][0]['gr']
                pos = gr.split("=")[0].split(",")[0]
                pos_counter_bk[pos] += 1 

    pos_tags = []
    counts = []

    for pos, count in pos_counter_bk.most_common():
        pos_tags.append(pos)
        counts.append(count)

    percents = [count / sum(counts) * 100 for count in counts]
    bk_plt = ax.bar(pos_tags, percents, )
    ax.set_title('TОП ЧАСТЕЙ РЕЧИ в %')
    plt.xticks(rotation=30)
    
    return bk_plt



def plot_top_names(tdf, ax):
    morph = pymorphy2.MorphAnalyzer()

    def names_extr(wrds):
        res = []
        for wrd in wrds:
            p = morph.parse(wrd)[0]
            if 'Name' in p.tag:
                res.append(wrd)
        return res

    text = str(tdf['text'].values)

    without_stop_words = tokenize(text, stop_words)
    without_stop_words = lemmatize(without_stop_words, morph)
    is_name = names_extr(without_stop_words)
    Fdist = FreqDist(is_name)

    df = pd.DataFrame(list(Fdist.items()), columns=['Name', 'Count']).sort_values(by = 'Count',ascending=False)
    df = df.head(20)

    top_names_plot = sns.barplot(x="Count",
                             y="Name",
                             data=df,
                             legend=False, ax=ax)
    ax.set_title('ТОП ИМЁН')

    return top_names_plot


def create_plot():
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18.5, 14), layout="constrained")
    return fig, axs


def plot_all_graphs(axs, html_content): #fig
    try:
        tdf = extract_data(html_content)
        if isinstance(tdf, pd.DataFrame)==False:
            if tdf==None: 
                print('tdf отсутствует')  
                return None
            
    except FileNotFoundError: 
        print("Файл пока нет.")
    

    messages=messages = [f'Обработано постов блога: {len(tdf)}', 
                         f"Среднее кол-во слов в предложении ~ {len_sent(tdf)},",
                         f"Доля уникальных слов составляет ~ {round(uniq_words_share(tdf,stop_words))}%",
                         f"Список 20-ти редко встречающихся слов: {process_and_visualize(stop_words, tdf)}"]

    plot_clean(tdf,stop_words, axs[0, 0])
    # Вызовите остальные функции и передайте им соответствующий объект ax
    plot_dirty(tdf, axs[0, 1])
    plot_pos(tdf, axs[1, 0])
    plot_top_names(tdf, axs[1, 1])
    #fig.tight_layout()
    
    return messages#, fig


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(18.5, 14), layout="constrained")


def exit(exitCode):
    print(exitCode)
    print(traceback.format_exc())


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}!')
    change_mode(message)

def change_mode(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Загрузить")
    btn2 = types.KeyboardButton("🛈 Инструкция")
    markup.add(btn1, btn2)
    
    bot.send_message(message.chat.id, f'Чем могу помочь?', reply_markup=markup)
    bot.register_next_step_handler(message, mode_router)

def mode_router(message):
    if message.text == 'Загрузить':
        search(message)
    elif message.text == "🛈 Инструкция":
        send_instruction(message)
    else:
        bot.send_message(message.chat.id, 'Неверный ввод. Выберите один из вариантов (кнопок).')
        change_mode(message)

def send_instruction(message):
    try:
        with open('source/инструкция тг.pdf', 'rb') as file:
            bot.send_document(message.chat.id, file)
        change_mode(message)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        exit("Failed to convert name" + str(e))
        bot.send_message(message.chat.id, 'Файл пока недоступен')
        change_mode(message)

def search(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True).add(types.KeyboardButton("🔙 Назад"))
    bot.send_message(message.chat.id, 'Отправьте мне выгрузку канала автора в виде html-файла', reply_markup=markup)
    bot.register_next_step_handler(message, step_1)

def step_1(message):
    if message.text == '🔙 Назад':
        change_mode(message)
        
    elif message.document and message.document.mime_type == 'text/html':
        process_html_file(message)
        
    else:
        bot.send_message(message.chat.id, 'Нужно выслать html-файл')
        bot.register_next_step_handler(message, step_1)

def process_html_file(message):
    try:
        file_path = bot.get_file(message.document.file_id).file_path

        # Загружаем файл из Telegram
        response = requests.get(f'https://api.telegram.org/file/bot{TOKEN}/{file_path}')
        html_content=response.content.decode('utf-8')
        print('Файл загружен в память')
        
        fig, axs = create_plot()  
        messages = plot_all_graphs(axs, html_content) 

        if not fig and not messages:
            bot.send_message(message.chat.id, 'Пусто')
            search(message)
            return

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        for text in messages:
            bot.send_message(message.chat.id, text)

        bot.send_photo(message.chat.id, buf)

        del html_content
        
        plt.clf()
        plt.close('all')
        change_mode(message)
        
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        exit("Failed to convert name" + str(e))
        bot.send_message(message.chat.id, 'Непредвиденная ошибка')
        search(message)

bot.infinity_polling()
