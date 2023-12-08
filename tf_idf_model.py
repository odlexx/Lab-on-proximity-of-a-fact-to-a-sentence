import re
import heapq
from math import log
import NLPmodule 
import pymorphy2
morph = pymorphy2.MorphAnalyzer()


#выбор, считаем tf как частоту (False), или логарифм (True)
TF_LOG_VALUE = True

with open("documents.txt", "r", encoding="UTF8") as f:
    text = f.read()

#разделение текста на предложения с помощью встроенной библиотеки
split_regex = re.compile(r'[.|!|?|…]')
#список документов (каждый документ - отдельное предложение)
documents = [t.strip() for t in split_regex.split(text)]

#список документов после обработки слов морфологическим анализатором, разделение по словам
list_of_docs = [[ morph.parse(NLPmodule.punctuation_marks_split(word))[0].normal_form  for word in doc.lower().split()] for doc in documents]

#создается словарь слово -- пара tf, df слов по всей коллекции
tf_df_dict = dict()
for doc in list_of_docs:
    for word in doc:
        tf_df_dict[word] = [tf_df_dict.get(word,[0,0])[0] + 1, tf_df_dict.get(word,[0,0])[1]] 
    for unique_word in set(doc):
        tf_df_dict[unique_word] = [tf_df_dict.get(unique_word,[0,0])[0], tf_df_dict.get(unique_word,[0,0])[1] + 1]

#мера tf заменяется на меру log(1 + tf)
if TF_LOG_VALUE:
    for key in tf_df_dict:
        tf_df_dict[key] = [log(tf_df_dict.get(key,[0,0])[0] + 1, 10), tf_df_dict.get(key,[0,0])[1]] 

#словарь всех слов       
vocabulary = list(tf_df_dict.keys())

#задаются и нормализуются запросы
init_queries = ["Жан Кокто написал 40 портретов мнимой немецкой кинозвезды",
                "Футболист остался после Первой мировой войны инвалидом, но продолжил спортивную карьеру",
                "Немецкая горнолыжница восемь раз попадала в пятёрку лучших на Олимпийских играх, но так и не стала чемпионкой"]
list_of_queries = [[ morph.parse(NLPmodule.punctuation_marks_split(word))[0].normal_form  for word in query.lower().split() ] for query in init_queries]

#создаются и заполняются вектора документов
doc_vectors = [[0 for i in range(len(vocabulary))] for doc in range(len(list_of_docs))]
j=0 #номер документа, по которому строим вектор
for doc in list_of_docs:
    for word in doc:
        doc_vectors[j][vocabulary.index(word)] += 1
    j+=1

#замена tf на log(tf + 1) для векторов документов. set накладываем, чтобы пройтись по каждому слову документа лишь 1 раз, избавляемся от повторов
if TF_LOG_VALUE:
    j=0 #номер документа, по которому строим вектор
    for doc in list_of_docs:
        for word in set(doc):
            doc_vectors[j][vocabulary.index(word)] = log(doc_vectors[j][vocabulary.index(word)] + 1, 10)
        j+=1

#домножаем веса слов на idf для документов
j=0 #номер документа, по которому строим вектор
for doc in list_of_docs:
    for word in doc:
        doc_vectors[j][vocabulary.index(word)] *= log( len(list_of_docs) / tf_df_dict[word][1], 10)
    j+=1   

#вычисляем вектора tf для запросов
query_vectors = [[0 for i in range(len(vocabulary))] for que in range(len(list_of_queries))]
j=0 #номер запроса, по которому строим вектор
for query in list_of_queries:
    for word in query:
        if word in vocabulary: query_vectors[j][vocabulary.index(word)] += 1
    j+=1

#замена tf на log(tf + 1) для векторов запросов
if TF_LOG_VALUE:
    j=0 #номер запроса, по которому строим вектор
    for query in list_of_queries:
        for word in set(query):
            if word in vocabulary: query_vectors[j][vocabulary.index(word)] = log( query_vectors[j][vocabulary.index(word)] + 1 , 10)
        j+=1    

#вычисление близости по косинусной мере
similarities = [[0 for i in list_of_docs] for j in list_of_queries]
for i in range(len(list_of_queries)):
    for j in range(len(list_of_docs)):
        similarities[i][j] = NLPmodule.measure_cos(query_vectors[i],doc_vectors[j])

#поиск индексов максимальных 10 значений и запись результата в файл
indecies_of_res = [[similarities[j].index(val) for val in heapq.nlargest(10,similarities[j])] for j in range(len(similarities))]
if TF_LOG_VALUE:
    res_filename = "result_logtf.txt"
else: res_filename = "result_tf.txt"
with open (res_filename, "w", encoding="UTF8") as ff:
    for i in range(len(indecies_of_res)):
        ff.write(init_queries[i] + "\n" )
        for j in range(len(indecies_of_res[0])):
            ff.write(str(similarities[i][indecies_of_res[i][j]]) + "\t   ")
            ff.write(documents[indecies_of_res[i][j]])
            ff.write("\n")
        ff.write("\n\n\n")
