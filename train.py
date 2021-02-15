import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout

from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intent_are.json').read())

words = []
classes = []
documents = []

ignore_letter = ['?','!','.',',']

for intent in intents['intents']:
    # print(intent)
    for pattern in intent['patterns']:
        # print(pattern) # all the pattern
        word_list = nltk.word_tokenize(pattern) # tokenize the words
        words.extend(word_list) # Extend help to add element in list (List will extend) length wise
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag']) #classes will have all the tag which are Present
# print(words)
# print(documents)
# print(classes)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letter] #takes into consideration the morphological analysis of the words.
words = sorted(set(words))
# print(words) #all the word in Patterns in sorted order

classes = sorted(set(classes))
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes) # Have 6 classes so [0, 0, 0, 0, 0, 0]
# print(output_empty)

for document in documents:
    # print(document) #(['Hi'], 'greeting')
    bag = []
    word_patterns = document[0]
    # print(word_patterns) # ['Hi']
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # print(word_patterns)  # words will be lemmatize and in lower case
    for word in words:
        # print(f"This is Word : {word}")
        # print(f"This is Word_Patterns : {word_patterns}")

        bag.append(1) if word in word_patterns else bag.append(0)
        # print(bag)

        output_row = list(output_empty)
        # print(document[1]) #greeting,goodbye and all the remaining tag
        # print(classes) #['age', 'goodbye', 'greeting', 'hours', 'name', 'shop']
        # print(classes.index(document[1]))
        output_row[classes.index(document[1])] = 1 #classes --> age and document[1] --> age == 1
        # print(f"Output Row: {output_row}")
        training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

# print(f"training[:,1] : {training[:,1]}")
# print(f"training[:,0] : {training[:,0]}")
train_x = list(training[:,0])
train_y = list(training[:,1])


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9,nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('chatbotmodel.h5',hist)
print("Done")




