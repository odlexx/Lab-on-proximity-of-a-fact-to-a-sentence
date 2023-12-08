import re
import heapq
import NLPmodule 
import pymorphy2
morph = pymorphy2.MorphAnalyzer()


LAMBDA = 0.5 #параметр языковой модели
EPSILON = 0.2 #параметр сглаживание, добавляется к частоте каждого слова

with open("documents.txt", "r", encoding="UTF8") as f:
    text = f.read()

#разделение текста на предложения с помощью встроенной библиотеки
split_regex = re.compile(r'[.|!|?|…]')
sentences = filter(lambda t: t, [t.strip() for t in split_regex.split(text)])

#список документов (каждый документ - отдельное предложение)
documents = list(sentences) 

#список документов после обработки слов морфологическим анализатором, разделение по словам
list_of_docs = [[ morph.parse(NLPmodule.punctuation_marks_split(word))[0].normal_form  for word in doc.lower().split()] for doc in documents]

#создается словарь слово -- частота
tf_dict = dict()
for doc in list_of_docs:
    for word in doc:
        tf_dict[word] = tf_dict.get(word,0) + 1

#задаются и нормализуются запросы
init_queries = ["Жан Кокто написал 40 портретов мнимой немецкой кинозвезды",
                "Футболист остался после Первой мировой войны инвалидом, но продолжил спортивную карьеру",
                "Немецкая горнолыжница восемь раз попадала в пятёрку лучших на Олимпийских играх, но так и не стала чемпионкой"]
list_of_queries = [[ morph.parse(NLPmodule.punctuation_marks_split(word))[0].normal_form  for word in query.lower().split() ] for query in init_queries]

#коллекция пополняется новыми словами из запросов
for i in range(len(list_of_queries)):
        for word in list_of_queries[i]:
            if tf_dict.get(word,0) == 0: tf_dict[word] = 1

#словарь слово -- вероятность
collection_prob_dict = dict()
voc_size = sum(tf_dict.values())
num_of_uniques = len(tf_dict)
for word in tf_dict:
    collection_prob_dict[word] = (tf_dict[word] + EPSILON) / (voc_size + EPSILON * num_of_uniques)


#функция для вычисления вероятности по запросу, документу и коллекции
def prob_query_doc_col(query, doc, collection_prob_dict):
    document_prob_dict = dict()
    for word in collection_prob_dict:
        document_prob_dict[word] = (doc.count(word) + EPSILON) / (len(doc) + len(collection_prob_dict) * EPSILON)
    prob_d, prob_c = 1, 1
    for word in query:
        prob_d *= document_prob_dict[word]
        prob_c *= collection_prob_dict[word]
    return LAMBDA * prob_d + (1-LAMBDA) * prob_c

#вычисление вероятностей
probabilities = [[0 for i in list_of_docs] for j in list_of_queries]
for i in range(len(list_of_queries)):
    for j in range(len(list_of_docs)):
        probabilities[i][j] = prob_query_doc_col(list_of_queries[i], list_of_docs[j],collection_prob_dict )

#вспомогательная функция для получения 10 больших элементов (иначе было некорректно для наличия одинаковых по величине значений)
def get_indecies_by_val(lst_of_val, ls):
    res = list()
    lst = list(ls)
    for val in lst_of_val:
        res.append(lst.index(val))
        lst[lst.index(val)] = 0 
    return res

#поиск индексов максимальных 10 значений и запись результата в файл
indecies_of_res = [get_indecies_by_val(heapq.nlargest(10,probabilities[j]), probabilities[j]) for j in range(len(probabilities))]
res_filename = f"result_{LAMBDA}.txt"
with open (res_filename, "w", encoding="UTF8") as ff:
    for i in range(len(indecies_of_res)):
        ff.write(init_queries[i] + "\n" )
        for j in range(len(indecies_of_res[0])):
            ff.write(str(probabilities[i][indecies_of_res[i][j]]) + "\t   ")
            ff.write(documents[indecies_of_res[i][j]])
            ff.write("\n")
        ff.write("\n\n\n")