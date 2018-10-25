from rx import Observable

file = open('lambtest.txt')

source = Observable.from_(file).flat_map(lambda s: s.split()).to_list().map(lambda x: len(x))

source.subscribe(lambda value: print("Reactive word count: {0}".format(value)))

file = open('lambtest.txt')
text = file.read()

words_list = text.split()

print ("Simple word count: {0}".format(len(words_list)))